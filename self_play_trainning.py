from pypokerengine.api.game import setup_config, start_poker
from bots.alpha0regretsbot import Alpha0Regret
from model import build_model

model = build_model(2)

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="p1", algorithm=Alpha0Regret(1,50,model,1))
config.register_player(name="p2", algorithm=Alpha0Regret(1,50,model,1))

game_result = start_poker(config, verbose=1)
print(game_result)