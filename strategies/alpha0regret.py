from pypokerengine.players import BasePokerPlayer
import numpy as np

class Alpha0regret(BasePokerPlayer):

    def __init__(self,model):
        super(Alpha0regret, self).__init__()
        self.model=model

    def set_action(self, action):
        self.action = action

    def declare_action(self, valid_actions, hole_card, round_state):
        pass
