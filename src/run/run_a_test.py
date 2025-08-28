from core.action import *
from core.game import MediumJong, Player
from core.learn.ac_constants import ACTION_HEAD_INDEX, TILE_HEAD_NOOP, chi_variant_index
from core.learn.policy_utils import encode_two_head_action
from core.tile import tile_flat_index


class LegalityCheckPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def act(self, gs):  # type: ignore[override]
        # Verify all legal moves align with masks; then pick a move.
        moves = gs.legal_moves()
        act_mask = gs.legal_action_mask()

        # Validate each option matches action mask bits
        for move in moves:
            action_idx, tile_idx = encode_two_head_action(move)
            tile_mask = gs.legal_tile_mask(action_idx)
            assert act_mask[action_idx] == 1
            assert tile_mask[tile_idx] == 1
        # Delegate move selection to base Player
        return super().act(gs)

    def react(self, gs, options):  # type: ignore[override]
        act_mask = gs.legal_action_mask()
        # Validate each option matches action mask bits
        for r in options:
            action_idx, tile_idx = encode_two_head_action(r)
            tile_mask = gs.legal_tile_mask(action_idx)
            assert act_mask[action_idx] == 1
            assert tile_mask[tile_idx] == 1
        # Delegate reaction selection to base Player
        return super().react(gs, options)

def run():
    # Seed for deterministic shuffling
    rng_seed = 0
    __import__('random').seed(rng_seed)

    g = MediumJong([
        LegalityCheckPlayer()
    ] * 4)
    # Drive the entire round; assertions inside the player will fail the test if inconsistent
    g.play_round(max_steps=10000)

if __name__ == '__main__':
    run()