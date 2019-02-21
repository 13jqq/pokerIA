import config
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Concatenate, average, LSTM, Masking, Softmax
from keras.optimizers import SGD
from keras import regularizers



def lr_scheduler(epochs):
    for k in config.training_param['LEARNING_RATE_SCHEDULE'].keys():
        if epochs >= k:
            return float(config.training_param['LEARNING_RATE_SCHEDULE'][k])

def actions_preprocessing(x, sharedLSTM):
    x=Masking(mask_value=0.0)(x)
    x=sharedLSTM(x)
    return x

def adv_preprocessing(x, sharedDense):
    x = sharedDense(x)
    x = BatchNormalization()(x)
    return x

def my_preprocessing(x, nb_unit):
    x = Dense(nb_unit,
              activation='relu',
              use_bias=True,
              kernel_initializer='glorot_normal',
              bias_initializer='zeros',
              kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
              name='my_preprocess'
              )(x)
    x = BatchNormalization()(x)
    return x

def model_hidden(x, nb_hidden, nb_hidden_unit):
    for i in range(nb_hidden):
        x=Dense(nb_hidden_unit,
                activation='relu',
                use_bias=True,
                kernel_initializer='glorot_normal',
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
                name='hidden_layer_'+str(i+1)
                )(x)
        x = BatchNormalization()(x)
    return x

def value_head(x, nb_unit):

    x = Dense(
        nb_unit,
        use_bias=False,
        activation='linear',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST'])
        )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Dense(
        1,
        use_bias=False,
        activation='tanh',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
        name = 'value_head'
        )(x)
    return (x)

def policy_head(x, nb_unit):
    x = Dense(
        nb_unit,
        use_bias=False,
        activation='linear',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST'])
    )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Dense(
        config.game_param['RAISE_PARTITION_NUM']+2,
        use_bias=False,
        activation='softmax',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
        name='policy_head'
    )(x)
    return (x)

def build_model(lstm_action_unit=config.network_param['LSTM_ACTION_PREPROC_UNIT'],
                player_sit_unit=config.network_param['PLAYER_SITUATION_PREPROC'],
                nb_hidden=config.network_param['NB_HIDDEN_LAYERS'],
                nb_hidden_unit=config.network_param['HIDDEN_LAYERS_UNITS'],
                nb_last_unit=config.network_param['LAST_LAYER_UNIT']):

    main_input = Input(shape=(16,), name = 'main_input')
    my_info = Input(shape=(3,), name = 'my_info')
    my_history = Input(shape=(None,7), name='my_history')
    adv_info = [Input(shape=(3,), name='adv_info_player' + str(i+1)) for i in range(config.game_param['MAX_PLAYER']-1)]
    adv_history=[Input(shape=(None,7), name='adv_history_player' + str(i+1)) for i in range(config.game_param['MAX_PLAYER']-1)]

    sharedLSTM = LSTM(lstm_action_unit,
                      activation='relu',
                      recurrent_activation='relu',
                      use_bias=True,
                      kernel_initializer='glorot_uniform',
                      recurrent_initializer='identity',
                      bias_initializer='zeros',
                      unit_forget_bias=True,
                      kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
                      recurrent_regularizer=None,
                      name='preprocess_action')

    sharedDense = Dense(player_sit_unit,
                        activation='relu',
                        use_bias=True,
                        kernel_initializer='glorot_normal',
                        bias_initializer='zeros',
                        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
                        name='adv_preprocess')

    x1 = actions_preprocessing(my_history, sharedLSTM)
    x1 = Concatenate(axis=-1)([main_input, my_info, x1])
    x1 = my_preprocessing(x1, player_sit_unit)

    x2=[]
    for idx,h in enumerate(adv_history):
        res = actions_preprocessing(h, sharedLSTM)
        res = Concatenate(axis=-1)([main_input, adv_info[idx], res])
        res = adv_preprocessing(res, sharedDense)
        x2.append(res)
    if len(x2) > 1:
        x2 = average(x2)
    else:
        x2 = x2[0]

    final_input = Concatenate(axis=-1)([x1,x2])

    x = model_hidden(final_input, nb_hidden, nb_hidden_unit)

    vh = value_head(x, nb_last_unit)
    ph = policy_head(x, nb_last_unit)

    model = Model(inputs=[main_input, my_info,my_history, *adv_info, *adv_history], outputs=[vh, ph])
    model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
        optimizer=SGD(lr=config.training_param['LEARNING_RATE_SCHEDULE'][0], momentum=config.training_param['MOMENTUM']),
        loss_weights={'value_head': 0.5, 'policy_head': 0.5})
    return model
