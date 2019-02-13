from collections import deque
import config


class Memory:
    def __init__(self):
        self.MEMORY_SIZE = config.training_param['MEMORY_SIZE']
        self.ltmemory = deque(maxlen=self.MEMORY_SIZE)
        self.stmemory = deque(maxlen=self.MEMORY_SIZE)

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

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=self.MEMORY_SIZE)
