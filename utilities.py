from pypokerengine.engine.action_checker import ActionChecker
import config

def build_action_list(stack,players):
    actions = [{"action": "fold", "amount": 0}, {"action": "call", "amount": ActionChecker().agree_amount(players)}]
    actions = actions + [{'action': 'raise', 'amount': int((x * stack) / config.game_param['RAISE_PARTITION_NUM'])} for
                         x in range(1, config.game_param['RAISE_PARTITION_NUM'] + 1)]
    return actions

def compare_action(action1,action2):
    res = False
    if action1['action'] == action2['action']:
        if isinstance(action1['amount'],int) and isinstance(action2['amount'],int):
            if action1['amount'] == action2['amount']:
                res=True
        elif isinstance(action1['amount'],dict) and isinstance(action2['amount'],int):
            if action2['amount']>=action1['amount']['min'] and  action2['amount']<=action1['amount']['max']:
                res=True
        elif isinstance(action2['amount'], dict) and isinstance(action1['amount'], int):
            if action1['amount'] >= action2['amount']['min'] and action1['amount'] <= action2['amount']['max']:
                res = True
        elif isinstance(action1['amount'],dict) and isinstance(action2['amount'],dict):
            if action1['amount']['min'] == action2['amount']['min'] and action1['amount']['max'] == action2['amount']['max']:
                res = True
    return res
