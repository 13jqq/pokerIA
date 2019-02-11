from pypokerengine.engine.action_checker import ActionChecker
import config
import numpy as np

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

def stable_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def parse_action(action,divider=1):
    id=[0] * len(config.game_param['ACTIONID'])
    id[config.game_param['ACTIONID'].index(action['action'])]=1
    amount=0
    try:
        amount=action['paid']
    except KeyError:
        try:
            amount = action['amount']
        except KeyError:
            pass
    return id + [(amount/divider)]

def merge_pkmn_dicts_same_key(ds):
    keys=list(set([k for d in ds for k in d.keys()]))
    res = {}
    [res.update({k: [d[k] for d in ds if k in d.keys()]}) for k in keys]
    return res

def truncate_float(num: float, n: int = 2) -> float:
    return int(num*10**n)/10**n
