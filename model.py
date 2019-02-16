import config
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Concatenate, average, LSTM, Masking, Softmax
from keras.optimizers import SGD
from keras import regularizers

sharedLSTM=LSTM(100,
           activation='relu',
           recurrent_activation='relu',
           use_bias = True,
           kernel_initializer = 'glorot_uniform',
           recurrent_initializer = 'identity',
           bias_initializer = 'zeros',
           unit_forget_bias = True,
           kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
           recurrent_regularizer=None,
           name='preprocess_action')

sharedDense = Dense(500,
                activation='relu',
                use_bias=True,
                kernel_initializer='glorot_normal',
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
                name='adv_preprocess')

pred_value_unit=config.game_param['MAX_PLAYER']
if pred_value_unit < 3:
    pred_value_unit = 1

def lr_scheduler(epochs):
    for k in config.training_param['LEARNING_RATE_SCHEDULE'].keys():
        if epochs > k:
            return config.training_param['LEARNING_RATE_SCHEDULE'][k]

def actions_preprocessing(x):
    x=Masking(mask_value=0.0)(x)
    x=sharedLSTM(x)
    return x

def adv_preprocessing(x):
    x = sharedDense(x)
    x = BatchNormalization()(x)
    return x

def my_preprocessing(x):
    x = Dense(500,
              activation='relu',
              use_bias=True,
              kernel_initializer='glorot_normal',
              bias_initializer='zeros',
              kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
              name='my_preprocess'
              )(x)
    x = BatchNormalization()(x)
    return x

def model_hidden(x):
    for i in range(config.network_param['NB_HIDDEN_LAYERS']):
        x=Dense(500,
                activation='relu',
                use_bias=True,
                kernel_initializer='glorot_normal',
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
                name='hidden_layer_'+str(i+1)
                )(x)
        x = BatchNormalization()(x)
    return x

def value_head(x):

    x = Dense(
        32,
        use_bias=False,
        activation='linear',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST'])
        )(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Dense(
        pred_value_unit,
        use_bias=False,
        activation='tanh',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
        name = 'value_head'
        )(x)
    return (x)

def policy_head(x):
    x = Dense(
        32,
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

def build_model():
    main_input = Input(shape=(16,), name = 'main_input')
    my_info = Input(shape=(3,), name = 'my_info')
    my_history = Input(shape=(None,7), name='my_history')
    adv_info = [Input(shape=(3,), name='adv_info_player' + str(i+1)) for i in range(config.game_param['MAX_PLAYER']-1)]
    adv_history=[Input(shape=(None,7), name='adv_history_player' + str(i+1)) for i in range(config.game_param['MAX_PLAYER']-1)]

    x1 = actions_preprocessing(my_history)
    x1 = Concatenate(axis=-1)([main_input, my_info, x1])
    x1 = my_preprocessing(x1)

    x2=[]
    for idx,h in enumerate(adv_history):
        res = actions_preprocessing(h)
        res = Concatenate(axis=-1)([main_input, adv_info[idx], res])
        res = adv_preprocessing(res)
        x2.append(res)
    if len(x2) > 1:
        x2 = average(x2)
    else:
        x2 = x2[0]

    final_input = Concatenate(axis=-1)([x1,x2])

    x = model_hidden(final_input)

    vh = value_head(x)
    ph = policy_head(x)

    model = Model(inputs=[main_input, my_info,my_history, *adv_info, *adv_history], outputs=[vh, ph])
    model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
        optimizer=SGD(lr=config.training_param['LEARNING_RATE_SCHEDULE'][0], momentum=config.training_param['MOMENTUM']),
        loss_weights={'value_head': 0.5, 'policy_head': 0.5})
    return model
