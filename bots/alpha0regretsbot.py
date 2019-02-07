from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from gamestate import GameState
import config
import math
import mccfr as mc
import numpy as np


class RLPLayer(BasePokerPlayer):

    def __init__(self, cpuct, mccfr_simulations, model):
        super(RLPLayer, self).__init__()
        self.cpuct = cpuct
        self.MCCFR_simulations = mccfr_simulations
        self.model = model
        self.mccfr = None

    # Setup Emulator object by registering game information
    def receive_game_start_message(self, game_info):
        player_num = game_info["player_num"]
        max_round = game_info["rule"]["max_round"]
        small_blind_amount = game_info["rule"]["small_blind_amount"]
        ante_amount = game_info["rule"]["ante"]
        blind_structure = game_info["rule"]["blind_structure"]

        self.emulator = Emulator()
        self.emulator.set_game_rule(player_num, max_round, small_blind_amount, ante_amount)
        self.emulator.set_blind_structure(blind_structure)

    def declare_action(self, valid_actions, hole_card, round_state):
        state=GameState(self.uuid, round_state, hole_card)
        if self.mccfr == None or state.id not in self.mccfr.tree:
            self.buildMCCFR(state)
        else:
            self.changeRootMCCFR(state)

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
        inputToModel = np.array([self.model.convertToModelInput(state)])

        preds = self.model.predict(inputToModel)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]

        logits = logits_array[0]

        allowedActions = state.allowedActions

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -math.inf

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)  ###put this just before the for?

        return ((value, probs, allowedActions))

    def getAV(self):
        edges = self.mccfr.root.edges
        pi = np.zeros(config.game_param['RAISE_PARTITION_NUM']+2, dtype=np.integer)

        for action, edge in edges:
            pi[action] = edge.stats['R']

        pi = pi / (np.sum(pi))
        return pi

    def build_allowed_action(self, allowed_action):
        pass
