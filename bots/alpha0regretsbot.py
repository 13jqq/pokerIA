from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.card_utils import gen_cards
from gamestate import GameState
from model import build_model
from utilities import stable_softmax, truncate_float, parse_action, merge_pkmn_dicts_same_key, treat_neg_regret
import config
import random
import argparse
import mccfr as mc
import numpy as np

#default bot parameters
num_player = 2
cpuct = 1
nb_mccfr_sim = config.mccfr['MCCFR_SIMS']
weight = None


class Alpha0Regret(BasePokerPlayer):

    def __init__(self, cpuct, mccfr_simulations, model, exp=0, uuid=None, emulator=None, memory=None):
        super(Alpha0Regret, self).__init__()
        self.cpuct = cpuct
        self.MCCFR_simulations = mccfr_simulations
        self.model = model
        self.mccfr = None
        self.intial_round_stack = 0
        self.total_round_money = 0
        self.emulator = emulator
        self.memory = memory
        if uuid is not None:
            self.uuid = uuid
        self.exp = exp

    # Setup Emulator object by registering game information
    def declare_action(self, valid_actions, hole_card, round_state):
        if self.total_round_money is None:
            pay_history = merge_pkmn_dicts_same_key(
                [{x['uuid']: parse_action(x, 1)[6]} for k in round_state['action_histories'].keys() for x
                 in round_state['action_histories'][k]])
            self.total_round_money = sum([player['stack'] + sum(pay_history[player['uuid']]) for player in round_state['seats']])

        if self.intial_round_stack is None:
            pay_history = merge_pkmn_dicts_same_key(
                [{x['uuid']: parse_action(x, 1)[6]} for k in round_state['action_histories'].keys() for x
                 in round_state['action_histories'][k]])
            self.intial_round_stack = sum(
                [player['stack'] + sum(pay_history[player['uuid']]) for player in round_state['seats'] if player['uuid'] == self.uuid])

        state=GameState(self.uuid, round_state, gen_cards(hole_card), self.emulator)
        if self.mccfr == None or state.id not in self.mccfr.tree:
            self.buildMCCFR(state)
        else:
            self.changeRootMCCFR(state)
        for sim in range(self.MCCFR_simulations):
            self.simulate()

        pi = self.getAV()

        action = self.chooseAction(pi)
        if self.memory is not None:
            self.memory.commit_stmemory(self.uuid,state.convertStateToModelInput(), pi, np.zeros((config.game_param['MAX_PLAYER'],)))

        return list(state.get_action_list()[action].values())

    def receive_game_start_message(self, game_info):
        player_num = game_info["player_num"]
        max_round = game_info["rule"]["max_round"]
        small_blind_amount = game_info["rule"]["small_blind_amount"]
        ante_amount = game_info["rule"]["ante"]
        blind_structure = game_info["rule"]["blind_structure"]

        self.emulator = Emulator()
        self.emulator.set_game_rule(player_num, max_round, small_blind_amount, ante_amount)
        self.emulator.set_blind_structure(blind_structure)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.intial_round_stack = None
        self.total_round_money = None

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        value = ([player['stack'] for player in round_state['seats'] if player['uuid'] == self.uuid][0] - self.intial_round_stack)/self.total_round_money


    def buildMCCFR(self, state):
        self.root = mc.Node(state)
        self.mccfr = mc.MCCFR(self.root, self.cpuct)

    def changeRootMCCFR(self, state):
        self.mccfr.root = self.mccfr.tree[state.id]

    def simulate(self):
        leaf, value, done, breadcrumbs = self.mccfr.moveToLeaf()
        value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)
        self.mccfr.backFill(value, breadcrumbs)

    def evaluateLeaf(self, leaf, value, done, breadcrumbs):
        if done == 0:

            value, probs, allowedActions = self.get_preds(leaf.state)
            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                newState, _, _ = leaf.state.takeAction(action)
                if newState.id not in self.mccfr.tree:
                    node = mc.Node(newState)
                    self.mccfr.addNode(node)
                else:
                    node = self.mccfr.tree[newState.id]
                newEdge = mc.Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))

        return ((value, breadcrumbs))

    def get_preds(self, state):
        # predict the leaf
        main_input, my_info, my_history, adv_info, adv_history = state.convertStateToModelInput()
        print(main_input.shape, my_info.shape, my_history.shape, [a.shape for a in adv_info], [a.shape for a in adv_history])
        preds = self.model.predict([main_input, my_info, my_history, *adv_info, *adv_history])
        value_array = preds[0]
        logits_array = preds[1]
        if config.game_param['MAX_PLAYER'] < 3:
            value = {x.uuid:(-1)**(idx)*value_array[0][0] for idx,x in enumerate(state.state['table'].seats.players)}
        else:
            value_sum_zero_array = np.asarray([value_array[0][idx] for idx, x in enumerate(state['table'].seats.players)])
            value_sum_zero_array = value_sum_zero_array - (sum(value_sum_zero_array)/len(value_sum_zero_array))
            value = {x.uuid: value_sum_zero_array[idx] for idx, x in enumerate(state['table'].seats.players)}

        logits = logits_array[0]

        allowedActions = state.allowed_action

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = float('-inf')

        probs = stable_softmax(logits)

        return ((value, probs, allowedActions))

    def getAV(self):
        edges = self.mccfr.root.edges
        pi = np.zeros(config.game_param['RAISE_PARTITION_NUM']+2, dtype=np.float32)
        for action, edge in edges:
            pi[action] = max(0, edge.stats['R'])
        if np.sum(pi) == 0:
            for action, edge in edges:
                pi[action] = treat_neg_regret(abs(min(0, edge.stats['R'])))

        pi = pi / np.sum(pi)
        return pi

    def chooseAction(self, pi):
        pi=np.asarray([truncate_float(x,7) for x in pi])
        if self.exp == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        return action

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nump', '-np', default=num_player, help='number of players')
    parser.add_argument('--cpuct', '-cp', default=cpuct, help='cpuct')
    parser.add_argument('--mccfr_simulations', '-mc', default=nb_mccfr_sim, help='Number of montecarlo simulation')
    parser.add_argument('--weight', '-w', default=weight, help='path for weights')
    return parser.parse_args()

def setup_ai():
    args = parse_args()
    model = build_model(args.nump)
    if args.weight is not None:
        model.load_weights(args.weight, by_name=True)
    return Alpha0Regret(args.cpuct, args.mccfr_simulations, model)


