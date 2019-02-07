import config
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Concatenate, average, LSTM, Masking
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


def actions_preprocessing(x):
    x=Masking(mask_value=0.0)(x)
    x=sharedLSTM(x)
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
        1,
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
        activation='sigmoid',
        kernel_initializer='glorot_normal',
        kernel_regularizer=regularizers.l2(config.training_param['REG_CONST']),
        name='policy_head'
    )(x)
    return (x)

def build_model(num_player):
    main_input = Input(shape=(14+(config.game_param['MAX_PLAYER']*3),), name = 'main_input')
    my_history = Input(shape=(None,7), name='my_history')
    adv_history=[Input(shape=(None,7), name='adv_history_player' + str(i+1)) for i in range(num_player-1)]

    x1 = actions_preprocessing(my_history)
    x2=[]
    for h in adv_history:
        x2.append(actions_preprocessing(h))
    if len(x2) > 1:
        x2 = average(x2)
    else:
        x2 = x2[0]

    final_input = Concatenate(axis=-1)([main_input,x1,x2])

    x = model_hidden(final_input)

    vh = value_head(x)
    ph = policy_head(x)

    model = Model(inputs=[main_input,my_history,*adv_history], outputs=[vh, ph])
    model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': 'categorical_crossentropy'},
        optimizer=SGD(lr=config.training_param['LEARNING_RATE'], momentum = config.training_param['MOMENTUM'],
                      decay=config.training_param['DECAY']),
        loss_weights={'value_head': 0.5, 'policy_head': 0.5}
        )
    return model
