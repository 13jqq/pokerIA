from pypokerengine.api.game import setup_config, start_poker
from bots.alpha0regretsbot import setup_ai
from bots.mcbot import MonteCarloBot
from bots.callbot import CallBot

config = setup_config(max_round=10, initial_stack=500, small_blind_amount=10)
config.register_player(name="p1", algorithm=setup_ai())
config.register_player(name="c1", algorithm=MonteCarloBot())

game_result = start_poker(config, verbose=1)

