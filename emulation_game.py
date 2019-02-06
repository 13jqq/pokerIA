from pypokerengine.api.emulator import Emulator
from gamestate import GameState
from pypokerengine.utils.card_utils import gen_cards
from pypokerengine.utils.game_state_utils import\
        restore_game_state, attach_hole_card, attach_hole_card_from_deck

def setup_game_state(uuid,round_state, my_hole_card):
    game_state = restore_game_state(round_state)
    for player_info in round_state['seats']:
        if uuid == player_info['uuid']:
            # Hole card of my player should be fixed. Because we know it.
            game_state = attach_hole_card(game_state, uuid, my_hole_card)
        else:
            # We don't know hole card of opponents. So attach them at random from deck.
            game_state = attach_hole_card_from_deck(game_state, uuid)
    return game_state

emulator = Emulator()
emulator2=Emulator()
emulator.set_game_rule(player_num=3, max_round=10, small_blind_amount=5, ante_amount=0)

# 2. Setup GameState object

players_info = {
    "uuid-1": {"name": "player1", "stack": 100},
    "uuid-2": {"name": "player2", "stack": 100}
}
game_state = emulator.generate_initial_game_state(players_info)

"""
game_state, events = emulator.start_new_round(game_state)
game_state, events = emulator.apply_action(game_state, "call", 10)


game_state = setup_game_state("uuid-1",events[-1]["round_state"], gen_cards(['DJ','HJ']))

paid_sum
print(game_state)

"""

game_state, events = emulator.start_new_round(game_state)
print(game_state)
print(events[-1]["round_state"])
game_state, events = emulator.apply_action(game_state, "raise", 20)
print(game_state)
print(events[-1]["round_state"])
test=GameState("uuid-1",events[-1]["round_state"], gen_cards(['DJ','HJ']))
print(test.id,test.model_input)
test2,value,done=test.takeAction({'action':"fold",'amount':0},emulator)
print(test2.id,value,done)


"""
game_state, events = emulator.apply_action(game_state, "raise", 20)
print([player.stack + player.paid_sum() for player in game_state['table'].seats.players])
print(events)
game_state, events = emulator.apply_action(game_state, "fold",0)
print([player.stack + player.paid_sum() for player in game_state['table'].seats.players])
print(events[-1]["round_state"])
game_state = setup_game_state("uuid-1",events[-1]["round_state"], gen_cards(['DJ','HJ']))
#game_state, events = emulator.start_new_round(game_state)
"""