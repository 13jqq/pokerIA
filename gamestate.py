from operator import itemgetter
from utilities import build_action_list, compare_action, parse_action, merge_pkmn_dicts_same_key
import itertools
import config
import numpy as np
from pypokerengine.utils.game_state_utils import\
        restore_game_state, attach_hole_card, attach_hole_card_from_deck
from pypokerengine.engine.action_checker import ActionChecker

class GameState():
    def __init__(self, my_uuid, round_state, my_hole_card, emulator):
        self.my_uuid=my_uuid
        self.my_hole_card = my_hole_card
        self.round_state = round_state
        self.state = self._setup_game_state(round_state)
        self.emulator = emulator
        self.playerTurn = None
        if self.state['next_player'] is not None and isinstance(self.state['next_player'],int):
            self.playerTurn = self.state['table'].seats.players[self.state['next_player']].uuid
        self.total_money = self._get_total_money()
        self.allowed_action = None
        if self.state['next_player'] is not None and isinstance(self.state['next_player'],int):
            self.allowed_action = self._get_allowed_action()
        self.id = self._convertStateToId()

    def _get_total_money(self):
        pay_history = merge_pkmn_dicts_same_key(
            [{x['uuid']: parse_action(x, 1)[6]} for k in self.round_state['action_histories'].keys() for x
             in self.round_state['action_histories'][k]])
        return sum([player['stack'] + sum(pay_history[player['uuid']]) for player in self.round_state['seats']])

    def _convertStateToId(self):
        hole_card = [card.__str__() for card in self.my_hole_card]
        community_card = [card.__str__() for card in self.state['table'].get_community_card()]
        try:
            blind_pos=[self.state['table'].sb_pos(),self.state['table'].bb_pos()]
        except Exception("blind position is not yet set"):
            blind_pos=[-1,-1]
        action_history = merge_pkmn_dicts_same_key(
            [{x['uuid']: [parse_action(x, self.total_money)]} for k in self.round_state['action_histories'].keys() for x
             in self.round_state['action_histories'][k]])
        my_info = list(itertools.chain.from_iterable([[idx, player.stack / self.total_money, int(player.is_active())] for
                   idx, player in enumerate(self.state['table'].seats.players) if player.uuid == self.my_uuid]))
        adv_info = sorted([[player.uuid, idx, player.stack / self.total_money, int(player.is_active())] for idx,player in enumerate(self.state['table'].seats.players) if
                           player.uuid != self.my_uuid], key=itemgetter(1))
        adv_order_list = [x[0] for x in adv_info]
        adv_info = list(itertools.chain.from_iterable([x[1:] for x in adv_info]))
        adv_history = list(itertools.chain.from_iterable([list(itertools.chain.from_iterable(action_history[key])) for key in adv_order_list if key != self.my_uuid]))
        my_history = list(itertools.chain.from_iterable(action_history[self.my_uuid]))

        stateId = hole_card + community_card + blind_pos + my_info + adv_info + my_history + adv_history
        id = ''.join(map(str, stateId))

        return id

    def _get_allowed_action(self):
        allowed_action_list=ActionChecker().legal_actions(self.state['table'].seats.players, self.state['next_player'],
                                      self.state['small_blind_amount'])
        action_list=build_action_list(self.state['table'].seats.players[self.state['next_player']].stack, self.state['table'].seats.players)
        res=[i for i,action in enumerate(action_list) for allowed_action in allowed_action_list if compare_action(action,allowed_action)]
        return list(set(res))

    def convertStateToModelInput(self):
        player=[player for player in self.state['table'].seats.players if player.uuid == self.playerTurn]
        if len(player)==1:
            hole_card=player[0].hole_card
        else:
            hole_card=self.my_hole_card
        hole_card = list(itertools.chain.from_iterable([[(card.suit-np.mean([2,4,8,16]))/np.std([2,4,8,16]),(card.rank-np.mean([2,3,4,5,6,7,8,9,10,11,12,13,14]))/np.std([2,3,4,5,6,7,8,9,10,11,12,13,14])] for card in hole_card]))
        community_card = list(itertools.chain.from_iterable([[(card.suit-np.mean([2,4,8,16]))/np.std([2,4,8,16]),(card.rank-np.mean([2,3,4,5,6,7,8,9,10,11,12,13,14]))/np.std([2,3,4,5,6,7,8,9,10,11,12,13,14])] for card in self.state['table'].get_community_card()]))
        cards = hole_card + community_card + ([0]*(14-len(hole_card) - len(community_card)))
        try:
            blind_pos=[self.state['table'].sb_pos(),self.state['table'].bb_pos()]
        except Exception("blind position is not yet set"):
            blind_pos=[-1,-1]
        action_history = merge_pkmn_dicts_same_key(
            [{x['uuid']: [parse_action(x, self.total_money)]} for k in self.round_state['action_histories'].keys() for x
             in self.round_state['action_histories'][k]])
        my_info=[[idx, player.stack / self.total_money, int(player.is_active())] for idx,player in enumerate(self.state['table'].seats.players) if player.uuid == self.playerTurn]
        adv_info=sorted([[player.uuid, idx, player.stack / self.total_money, int(player.is_active())] for idx, player in enumerate(self.state['table'].seats.players) if player.uuid != self.playerTurn], key=itemgetter(1))
        adv_order_list=[x[0] for x in adv_info]
        my_info = np.expand_dims(np.asarray(my_info[0]), axis=0)
        adv_info = [np.expand_dims(np.asarray(x[1:]), axis=0) for x in adv_info]
        main_input=np.expand_dims(np.asarray(cards+blind_pos), axis=0)
        my_history=np.expand_dims(np.asarray(action_history[self.playerTurn]).reshape(-1,7), axis=0)
        adv_history=[np.expand_dims(np.asarray(action_history[key]).reshape(-1,7), axis=0) for key in adv_order_list if key!=self.playerTurn]
        return main_input, my_info, my_history, adv_info, adv_history

    def _setup_game_state(self, round_state):
        game_state = restore_game_state(round_state)
        for player in game_state['table'].seats.players:
            if self.my_uuid == player.uuid:
                game_state = attach_hole_card(game_state, player.uuid, self.my_hole_card)
            else:
                game_state = attach_hole_card_from_deck(game_state, player.uuid)
        return game_state

    def get_action_list(self):
        return build_action_list(self.state['table'].seats.players[self.state['next_player']].stack, self.state['table'].seats.players)

    def takeAction(self, action):
        action = build_action_list(self.state['table'].seats.players[self.state['next_player']].stack, self.state['table'].seats.players)[action]
        game_state, events = self.emulator.apply_action(self.state, action['action'], action['amount'])
        value = {}
        done = 0
        if events[-1]['type'] == "event_game_finish":
            events.pop(-1)
        newState = GameState(self.my_uuid, events[-1]['round_state'], self.my_hole_card, self.emulator)
        if events[-1]['type'] == "event_round_finish":
            pay_history = merge_pkmn_dicts_same_key(
                [{x['uuid']: parse_action(x, 1)[6]} for k in self.round_state['action_histories'].keys() for x
                 in self.round_state['action_histories'][k]])
            value = {player2.uuid: (player2.stack - player.stack - sum(pay_history[player.uuid]))/self.total_money for player2 in
                     newState.state['table'].seats.players for player in
                     self.state['table'].seats.players if player.uuid == player2.uuid}
            done = 1

        return (newState, value, done)
