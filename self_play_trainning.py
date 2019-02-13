from pypokerengine.api.emulator import Emulator
from bots.alpha0regretsbot import Alpha0Regret
from model import build_model
from memory import Memory
from utilities import initialize_new_emulator, merge_pkmn_dicts_same_key, parse_action, to_list
import random
import config
import itertools
import os

foldername = list(itertools.chain.from_iterable([to_list(config.game_param[k]) for k in config.game_param.keys()])) + list(itertools.chain.from_iterable([to_list(config.network_param[k]) for k in config.network_param.keys()]))
foldername = "_".join([str(x) for x in foldername])

log_folder=os.path.join(config.training_param['LOG_DIR'], foldername)
savemodel=os.path.join(config.valuation_param['LAST_MODEL_DIR'], foldername)

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

if not os.path.exists(savemodel):
    os.makedirs(savemodel)

memory=Memory()


emulator = Emulator()
nb_player = random.randint(2, config.game_param['MAX_PLAYER'])
sb_amount = random.randint(1, 10)
ante_amount = random.randint(0, 5)
max_round=random.randint(10, 20)
model = build_model(nb_player)

emulator.set_game_rule(player_num=nb_player, max_round=max_round, small_blind_amount=sb_amount, ante_amount=ante_amount)
players_info={"uuid-"+str(i): {'name': 'player'+str(i), 'stack': random.randint(80, 120)} for i in range(0,nb_player)}


initial_state = emulator.generate_initial_game_state(players_info)
[emulator.register_player(k,Alpha0Regret(1,config.mccfr['MCCFR_SIMS'],model,1,k,initialize_new_emulator(nb_player, max_round, sb_amount, ante_amount), memory)) for k in players_info.keys()]
game_state=initial_state
events=[{'type': None}]
while events[-1]['type'] != "event_game_finish":
    game_state, events = emulator.start_new_round(game_state)
    if events[-1]['type'] != "event_game_finish":
        pay_history = merge_pkmn_dicts_same_key(
                [{x['uuid']: parse_action(x, 1)[6]} for k in events[-1]['round_state']['action_histories'].keys() for x
                 in events[-1]['round_state']['action_histories'][k]])
        stack_history = {x['uuid']: x['stack'] for x in events[-1]['round_state']['seats']}
        player_initial_stack = {k: stack_history[k] + sum(pay_history[k]) for k in stack_history.keys()}
        total_round_money = sum([player_initial_stack[k] for k in player_initial_stack.keys()])
        game_state, events = emulator.run_until_round_finish(game_state)
        if events[-1]['type'] == "event_game_finish":
            score = {x['uuid']: ((x['stack'] - player_initial_stack[x['uuid']]) / total_round_money) for x in events[-1]['players']}
        else:
            score = {x['uuid']: ((x['stack'] - player_initial_stack[x['uuid']]) / total_round_money) for x in
                     events[-1]['round_state']['seats']}
        for e in memory.stmemory:
            scorelist = [score[e['playerTurn']]] + [score[s] for s in score.keys() if s != e['playerTurn']]
            for i in range(0,len(scorelist)):
                e['score'][i] = scorelist[i]
        memory.commit_ltmemory()
print(memory.ltmemory)


