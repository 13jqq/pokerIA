from operator import itemgetter
import itertools
import numpy as np
import config
from pypokerengine.utils.game_state_utils import\
        restore_game_state, attach_hole_card, attach_hole_card_from_deck
from pypokerengine.engine.action_checker import ActionChecker

def parse_action(action,divider=1):
    id=[0] * len(config.game_param['ACTIONID'])
    id[config.game_param['ACTIONID'].index(action['action'])]=1
    amount=0
    try:
        amount=action['paid']
    except KeyError:
        try:
            amount = action['amount']
        except KeyError:
            pass
    return id + [(amount/divider)]


class GameState():
    def __init__(self, my_uuid, round_state, my_hole_card):
        self.my_uuid=my_uuid
        self.my_hole_card = my_hole_card
        self.state = self._setup_game_state(round_state)
        self.playerTurn = None
        if self.state['next_player'] is not None:
            self.playerTurn = self.state['table'].seats.players[self.state['next_player']].uuid
        self.total_money = sum([player.stack + player.paid_sum() for player in self.state['table'].seats.players])
        self.allowed_action = None
        if self.state['next_player'] is not None:
            self.allowed_action = ActionChecker().legal_actions(self.state['table'].seats.players,self.state['next_player'],self.state['small_blind_amount'])
        self.id = self._convertStateToId()
        self.model_input=self._convertStateToModelInput()

    def _convertStateToId(self):
        hole_card = [card.__str__() for card in self.my_hole_card]
        community_card = [card.__str__() for card in self.state['table'].get_community_card()]
        my_stack = list(itertools.chain.from_iterable(
            [[player.stack / self.total_money, int(player.is_active()), player.paid_sum() / self.total_money] +
             list(itertools.chain.from_iterable(
                 [parse_action(action, self.total_money) for action in player.action_histories])) for player in
             self.state['table'].seats.players if player.uuid == self.my_uuid]))
        op_stack = list(itertools.chain.from_iterable(
            sorted([[player.stack / self.total_money, int(player.is_active()), player.paid_sum() / self.total_money] +
                    list(itertools.chain.from_iterable(
                        [parse_action(action, self.total_money) for action in player.action_histories])) for player in
                    self.state['table'].seats.players if player.uuid != self.my_uuid], key=itemgetter(0))))
        stateId = hole_card + community_card + my_stack + op_stack
        id = ''.join(map(str, stateId))

        return id

    def _convertStateToModelInput(self):
        hole_card = list(itertools.chain.from_iterable([[(card.suit-np.mean([2,4,8,16]))/np.std([2,4,8,16]),(card.rank-np.mean([2,3,4,5,6,7,8,9,10,11,12,13,14]))/np.std([2,3,4,5,6,7,8,9,10,11,12,13,14])] for card in self.my_hole_card]))
        community_card = list(itertools.chain.from_iterable([[(card.suit-np.mean([2,4,8,16]))/np.std([2,4,8,16]),(card.rank-np.mean([2,3,4,5,6,7,8,9,10,11,12,13,14]))/np.std([2,3,4,5,6,7,8,9,10,11,12,13,14])] for card in self.state['table'].get_community_card()]))
        cards = hole_card+community_card + ([0]*(14-len(hole_card) - len(community_card)))


        input_model= cards
        return(input_model)

    def _setup_game_state(self, round_state):
        game_state = restore_game_state(round_state)
        for player in game_state['table'].seats.players:
            if self.my_uuid == player.uuid:
                game_state = attach_hole_card(game_state, player.uuid, self.my_hole_card)
            else:
                game_state = attach_hole_card_from_deck(game_state, player.uuid)
        return game_state

    def takeAction(self, action, emulator):

        game_state, events = emulator.apply_action(self.state, action['action'], action['amount'])
        value = {self.playerTurn: 0}
        done = 0
        newState = GameState(self.my_uuid, events[-1]['round_state'], self.my_hole_card)
        if events[-1]['type']=="event_round_finish":
            ante = emulator.game_rule["ante"]
            value = {player2.uuid: (player2.stack - player.stack - player.paid_sum() - ante)/self.total_money for player2 in
                     newState.state['table'].seats.players for player in
                     self.state['table'].seats.players if player.uuid == player2.uuid}
            done = 1

        return (newState, value, done)
