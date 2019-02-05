import config
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Concatenate, LSTM
from keras.optimizers import SGD
from keras import regularizers
import tensorflow as tf

def model_preprocessing(x):
    x=LSTM(100)(x)
    return x

def model_body(x):
    x=Dense(500,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_normal',
            bias_initializer='zeros',
            kernel_regularizer=regularizers.l2(config.training_param['REG_CONST'])
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
        name='value_head'
    )(x)
    return (x)

def build_model():
    main_input = Input(shape = config.game_param['MAX_PLAYER'], name = 'main_input')

    x = model_body(main_input)
    for i in range(config.network_param['NB_HIDDEN_LAYERS']-1):
        x = model_body(x)

    vh = value_head(x)
    ph = policy_head(x)

    model = Model(inputs=[main_input], outputs=[vh, ph])
    model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': tf.nn.softmax_cross_entropy_with_logits},
        optimizer=SGD(lr=config.training_param['LEARNING_RATE'], momentum = config.training_param['MOMENTUM'],
                      decay=config.training_param['DECAY']),
        loss_weights={'value_head': 0.5, 'policy_head': 0.5}
        )
    return model
