import random
import numpy as np

from src.core.game import MediumJong, Player


def test_game_outcome_uses_public_identifiers():
    # Deterministic seed to make the test stable
    random.seed(123)
    np.random.seed(123)

    # Create 4 players with high identifiers (> 4)
    ids = [10, 11, 12, 13]
    players = [Player(identifier=i) for i in ids]

    # Run a full game
    game = MediumJong(players)
    while not game.is_game_over():
        game.play_turn()

    outcome = game.get_game_outcome()

    # Collect the player_ids stored in the GameOutcome (these should be public identifiers)
    outcome_pub_ids = sorted(po.player_id for po in outcome.players.values())
    assert sorted(ids) == outcome_pub_ids, (
        f"Expected GameOutcome player_ids {sorted(ids)} but got {outcome_pub_ids}"
    )
