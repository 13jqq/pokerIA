game_param={
    'MAX_PLAYER': 10,
    'RAISE_PARTITION_NUM': 10

}

ismcts_sim={
    'EPISODES': 30,
    'MCTS_SIMS': 50,
    'MEMORY_SIZE': 30000
}

training_param={
    'BATCH_SIZE': 256,
    'EPOCHS': 1,
    'REG_CONST': 0.0001,
    'LEARNING_RATE': 0.1,
    'MOMENTUM': 0.9,
    'DECAY': 0.0
}

valuation_param={
    'EVAL_EPISODES': 20,
    'SCORING_THRESHOLD': 1.3
}

network_param={
    'NB_HIDDEN_LAYERS': 7
}