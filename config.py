game_param={
    'MAX_PLAYER': 2,
    'RAISE_PARTITION_NUM': 100,
    'ACTIONID' : ['ANTE','SMALLBLIND','BIGBLIND','FOLD','CALL','RAISE']
}

mccfr={
    'MCCFR_SIMS': 800,
    'EPSILON': 0.25,
    'ALPHA': 0.3,
    'CPUCT_BASE': 19652,
    'CPUCT_INIT': 1.25
}

training_param={
    'LOG_DIR':'D:/projetsIA/pokerIA/training_weights',
    'MEMORY_SIZE': 30000,
    'BATCH_SIZE': 256,
    'EPOCHS': 1,
    'REG_CONST': 0.0001,
    'MOMENTUM': 0.9,
    'LEARNING_RATE_SCHEDULE':  {
        0: 0.2,
        100000: 0.02,
        300000: 0.002,
        500000: 0.0002
    }
}

valuation_param={
    'LAST_MODEL_DIR': 'D:/projetsIA/pokerIA/final_weights'
}

network_param={
    'NB_HIDDEN_LAYERS': 1
}
