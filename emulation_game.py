from pypokerengine.api.emulator import Emulator
from strategies.random_choice import RandomModel
from pypokerengine.utils.game_state_utils import\
        restore_game_state, attach_hole_card, attach_hole_card_from_deck

def setup_game_state(uuid,round_state, my_hole_card):
    game_state = restore_game_state(round_state)
    for player_info in round_state['seats']:
        if uuid == player_info.uuid:
            # Hole card of my player should be fixed. Because we know it.
            game_state = attach_hole_card(game_state, uuid, my_hole_card)
        else:
            # We don't know hole card of opponents. So attach them at random from deck.
            game_state = attach_hole_card_from_deck(game_state, uuid)

def toId_state(round_state, my_hole_card):
    pass

emulator = Emulator()
emulator.set_game_rule(player_num=3, max_round=10, small_blind_amount=5, ante_amount=0)
# 2. Setup GameState object
players_info = {
    "uuid-1": {"name": "player1", "stack": 100},
    "uuid-2": {"name": "player2", "stack": 100}
}
initial_state = emulator.generate_initial_game_state(players_info)
emulator.register_player("uuid-1", RandomModel())
emulator.register_player("uuid-2", RandomModel())
game_state, events = emulator.start_new_round(initial_state)
print(game_state['table'].seats.players[0].hole_card[0].__str__())
print(events)
game_state, events = emulator.apply_action(game_state, "call", 10)
print(events)
game_state, events = emulator.apply_action(game_state, "raise", 20)
print(events)
game_state, events = emulator.apply_action(game_state, "call", 20)
print(events)
#game_state, events = emulator.start_new_round(game_state)


