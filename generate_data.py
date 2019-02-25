from pypokerengine.api.emulator import Emulator
from bots.alpha0regretsbot import Alpha0Regret
from model import build_model
from memory import Memory
from utilities import initialize_new_emulator, merge_pkmn_dicts_same_key, parse_action, to_list
import random
import shortuuid
import config
import itertools
import os

num_game = 10000

foldername = list(itertools.chain.from_iterable([to_list(config.game_param[k]) for k in config.game_param.keys()])) + list(itertools.chain.from_iterable([to_list(config.network_param[k]) for k in config.network_param.keys()]))
foldername = "_".join([str(x) for x in foldername])

data_folder = os.path.join(config.acquisition_param['ACQUISITION_DIR'], foldername)
log_folder_weights = os.path.join(config.training_param['LOG_DIR'], foldername,'weights')



if not os.path.exists(data_folder):
    os.makedirs(data_folder)

memory = Memory()
emulator = Emulator()
model = build_model()
starting_game = 0
if os.path.exists(log_folder_weights):
    log_weights = [f for f in os.listdir(log_folder_weights) if f.endswith('.h5')]
    if len(log_weights) > 0:
        log_weights.sort(key=lambda f: int(''.join(filter(str.isdigit, f))) or -1)
        model.load_weights(os.path.join(log_folder_weights,log_weights[-1]))

if os.path.exists(os.path.join(data_folder,'latest_memory.json')):
    memory.load_lt_memory(data_folder,'latest_memory.json')

for game in range(starting_game, starting_game + num_game):

    print(game)

    nb_player = random.randint(2, config.game_param['MAX_PLAYER'])
    sb_amount = random.randint(1, 10)
    ante_amount = random.randint(0, 5)
    max_round = random.randint(10, 20)

    emulator.set_game_rule(player_num=nb_player, max_round=max_round, small_blind_amount=sb_amount, ante_amount=ante_amount)
    players_info={"uuid-"+str(i): {'name': 'player'+str(i), 'stack': random.randint(80, 120)} for i in range(0,nb_player)}

    initial_state = emulator.generate_initial_game_state(players_info)
    [emulator.register_player(k,Alpha0Regret(100,model,1,k,initialize_new_emulator(nb_player, max_round, sb_amount, ante_amount), memory)) for k in players_info.keys()]
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
                if config.game_param['MAX_PLAYER'] < 3:
                    scorelist = [score[e['playerTurn']]]
                else:
                    scorelist = [score[e['playerTurn']]] + [score[s] for s in score.keys() if s != e['playerTurn']]
                for i in range(0,len(scorelist)):
                    e['score'][i] = scorelist[i]
            if (len(memory.ltmemory) + len(memory.stmemory)) > memory.MEMORY_SIZE:
                memory.save_lt_memory(data_folder, shortuuid.uuid() + '.json')
                memory.clear_ltmemory()
            memory.commit_ltmemory()
    memory.save_lt_memory(data_folder, 'latest_memory.json')


memory.save_lt_memory(data_folder,'latest_memory.json')



