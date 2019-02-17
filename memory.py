from collections import deque
import numpy as np
import config
import json
import os


class Memory:
    def __init__(self, path=None, filename=None):
        self.MEMORY_SIZE = config.training_param['MEMORY_SIZE']
        self.ltmemory = deque(maxlen=self.MEMORY_SIZE)
        self.stmemory = deque(maxlen=self.MEMORY_SIZE)
        if path is not None and filename is not None:
            if os.path.exists(os.path.join(path, filename)):
                self.load_lt_memory(path, filename)

    def commit_stmemory(self, uuid, input_state, output_probs, output_score):
        self.stmemory.append({
            'playerTurn': uuid,
            'nn_input': input_state,
            'probs': output_probs,
            'score': output_score
        })

    def commit_ltmemory(self):
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def commit_and_save_ltmemory(self, path, filename):
        if len(self.stmemory) + len(self.ltmemory) > self.MEMORY_SIZE:
            self.save_lt_memory(path, filename)
            self.clear_ltmemory()
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=self.MEMORY_SIZE)

    def clear_ltmemory(self):
        self.ltmemory = deque(maxlen=self.MEMORY_SIZE)

    def save_lt_memory(self, path, filename):
       save = [{'playerTurn': e['playerTurn'],
         'nn_input': [e['nn_input'][0].tolist(), e['nn_input'][1].tolist(), e['nn_input'][2].tolist(),
                      [x.tolist() for x in e['nn_input'][3]], [x.tolist() for x in e['nn_input'][4]]],
         'probs': e['probs'].tolist(), 'score': e['score'].tolist()} for e in
        self.ltmemory]
       with open(os.path.join(path, filename), 'w') as outfile:
            json.dump(save, outfile)

    def load_lt_memory(self, path, filename):
        with open(os.path.join(path, filename), 'r') as outfile:
            data = json.load(outfile)
        data = [{'playerTurn': e['playerTurn'],
         'nn_input': [np.asarray(e['nn_input'][0]), np.asarray(e['nn_input'][1]), np.asarray(e['nn_input'][2]),
                      [np.asarray(x) for x in e['nn_input'][3]], [np.asarray(x) for x in e['nn_input'][4]]],
         'probs': np.asarray(e['probs']), 'score': np.asarray(e['score'])} for e in data]
        self.ltmemory = deque(data, maxlen=self.MEMORY_SIZE)


    def convertToModelData(self, memoryType = 'st'):
        assert memoryType in ['st', 'lt'], 'choose between long term (lt) and short term (st) memory'
        if memoryType == 'st':
            currentMem = self.stmemory
        elif memoryType == 'lt':
            currentMem = self.ltmemory
        main_input = np.concatenate([e['nn_input'][0] for e in currentMem], axis=0)
        my_info = np.concatenate([e['nn_input'][1] for e in currentMem], axis=0)
        my_history_max_len = max([e['nn_input'][2].shape[1] for e in currentMem])
        my_history = np.concatenate([np.concatenate([e['nn_input'][2], np.zeros(
            (1, my_history_max_len - e['nn_input'][2].shape[1], e['nn_input'][2].shape[2]))], axis=1) for e in
                               currentMem], axis=0)
        adv_info = [np.concatenate([e['nn_input'][3][i] for e in currentMem], axis=0) for i in range(0,config.game_param['MAX_PLAYER'] - 1)]
        adv_history_max_len = [max([e['nn_input'][4][i].shape[1] for e in currentMem]) for i in range(0,config.game_param['MAX_PLAYER'] - 1)]
        adv_history = [np.concatenate([np.concatenate([e['nn_input'][4][i], np.zeros(
            (1, adv_history_max_len[i] - e['nn_input'][4][i].shape[1], e['nn_input'][4][i].shape[2]))], axis=1) for e in
                               currentMem], axis=0) for i in range(0,config.game_param['MAX_PLAYER'] - 1)]
        probs = np.stack([e['probs'] for e in currentMem], axis=0)
        score = np.stack([e['score'] for e in currentMem], axis=0)
        return {'input': [main_input, my_info, my_history, *adv_info, *adv_history], 'output': [score, probs]}
