game_param={
    'MAX_PLAYER': 2,
    'RAISE_PARTITION_NUM': 100,
    'ACTIONID' : ['ANTE','SMALLBLIND','BIGBLIND','FOLD','CALL','RAISE']
}

mccfr={
    'EPISODES': 30,
    'MCCFR_SIMS': 100,
    'EPSILON': 0.2,
    'ALPHA': 0.8
}

training_param={
    'LOG_DIR':'D:/projetsIA/pokerIA/training_weights',
    'MEMORY_SIZE': 30000,
    'BATCH_SIZE': 256,
    'EPOCHS': 1,
    'REG_CONST': 0.0001,
    'LEARNING_RATE': 0.1,
    'MOMENTUM': 0.9,
    'DECAY': 0.0
}

valuation_param={
    'LAST_MODEL_DIR': 'D:/projetsIA/pokerIA/final_weights',
    'EVAL_EPISODES': 20,
    'SCORING_THRESHOLD': 1.3
}

network_param={
    'NB_HIDDEN_LAYERS': 1
}
