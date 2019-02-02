from pypokerengine.players import BasePokerPlayer
import numpy as np
import itertools
import config

class RandomModel(BasePokerPlayer):

    def set_action(self, action):
        self.action = action

    def declare_action(self, valid_actions, hole_card, round_state):
        print(valid_actions)
        actions1 = [item for item in valid_actions if item['action']!='raise']
        action2=[ list(range(item['amount']['min'],item['amount']['max'],int((item['amount']['max']-item['amount']['min'])
                             /config.game_param['RAISE_PARTITION_NUM']))) + [item['amount']['max']] for item in
                       valid_actions if item['action']=='raise']
        action2=[{'action':'raise','amount':item} for item in list(itertools.chain.from_iterable(action2))]
        actions=actions1+action2
        print(actions)
        return list(np.random.choice(actions).values())
