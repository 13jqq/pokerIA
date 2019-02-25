from hyperopt import STATUS_OK, fmin, Trials, tpe, hp
from utilities import to_list
from model import build_model
from memory import Memory
import config
import os
import itertools
import numpy as np


MAX_EVALS = 10
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
    'LSTM_ACTION_PREPROC_UNIT': hp.qloguniform('LSTM_ACTION_PREPROC_UNIT', np.log(100), np.log(10000)),
    'PLAYER_SITUATION_PREPROC_UNIT': hp.qloguniform('PLAYER_SITUATION_PREPROC_UNIT', np.log(100), np.log(10000)),
    'NB_HIDDEN_LAYERS': hp.quniform('NB_HIDDEN_LAYERS', 1, 10),
    'HIDDEN_LAYERS_UNITS': hp.qloguniform('HIDDEN_LAYERS_UNITS', np.log(100), np.log(10000)),
    'LAST_LAYER_UNIT': hp.quniform('LAST_LAYER_UNIT', 1, 100),
    'LEARNING_RATE': hp.loguniform('LEARNING_RATE', np.log(0.01), np.log(0.3)),
    'MOMENTUM': hp.uniform('MOMENTUM', 0.0, 1.0),
    'REG_CONST': hp.loguniform('REG_CONST', np.log(0.00001), np.log(0.01)),
    'BATCH_SIZE': hp.choice('BATCH_SIZE',[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384])
}


trials = Trials()

best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = trials)
