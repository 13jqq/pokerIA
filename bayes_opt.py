from hyperopt import STATUS_OK, fmin, Trials, tpe, hp
from utilities import to_list
from model import build_model
from memory import Memory
import config
import os
import itertools

MAX_EVALS = 500
epoch = 100000
mem_limit = 100000


foldername = list(itertools.chain.from_iterable([to_list(config.game_param[k]) for k in config.game_param.keys()])) + list(itertools.chain.from_iterable([to_list(config.network_param[k]) for k in config.network_param.keys()]))
foldername = "_".join([str(x) for x in foldername])

data_folder = os.path.join(config.acquisition_param['ACQUISITION_DIR'], foldername)
memory = Memory(memory_size=min(len([file for file in os.listdir(data_folder) if file.endswith('.json')])*config.training_param['MEMORY_SIZE'], mem_limit))
[memory.load_lt_memory(data_folder,file) for file in os.listdir(data_folder) if file.endswith('.json')]

def objective(params):

    model = build_model(lstm_action_unit=params['LSTM_ACTION_PREPROC_UNIT'],
                player_sit_unit=params['PLAYER_SITUATION_PREPROC_UNIT'],
                nb_hidden=params['NB_HIDDEN_LAYERS'],
                nb_hidden_unit=params['HIDDEN_LAYERS_UNITS'],
                nb_last_unit=params['LAST_LAYER_UNIT'],
                lr=params['LEARNING_RATE'],
                momentum=params['MOMENTUM'],
                reg_const=params['REG_CONST'])
    data = memory.convertToModelData('lt')
    hist = model.fit(x=data['input'], y=data['output'], batch_size=params['BATCH_SIZE'],
              epochs=epoch, validation_split=0.2)

    return {'loss': hist.history['val_loss'][-1], 'params': params, 'status': STATUS_OK}

space = {
    'LSTM_ACTION_PREPROC_UNIT': ,
    'PLAYER_SITUATION_PREPROC_UNIT': ,
    'NB_HIDDEN_LAYERS': ,
    'HIDDEN_LAYERS_UNITS': ,
    'LAST_LAYER_UNIT': ,
    'LEARNING_RATE': ,
    'MOMENTUM': ,
    'REG_CONST': ,
    'BATCH_SIZE': ,
}

trials = Trials()

best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = trials)
