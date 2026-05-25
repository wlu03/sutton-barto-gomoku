import numpy as np
import pytest

from gomoku.env import BLACK, EMPTY, WHITE, GomokuEnv


def test_reset_state():
    env = GomokuEnv(size=9)
    s = env.reset()
    assert s.to_play == BLACK
    assert not s.terminal
    assert (s.board == EMPTY).all()


def test_alternating_turns():
    env = GomokuEnv(size=9)
    env.reset()
    s1 = env.step(0)
    assert s1.to_play == WHITE
    s2 = env.step(1)
    assert s2.to_play == BLACK


def test_illegal_move_raises():
    env = GomokuEnv(size=9)
    env.reset()
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)


def test_horizontal_win():
    env = GomokuEnv(size=9)
    env.reset()
    # Black plays row 0 cols 0..4, White plays row 1 cols 0..3.
    moves = [0, 9, 1, 10, 2, 11, 3, 12, 4]  # black wins on the 5th
    for m in moves[:-1]:
        s = env.step(m)
        assert not s.terminal
    s = env.step(moves[-1])
    assert s.terminal
    assert s.winner == BLACK
    assert s.reward == 1.0


def test_legal_mask_matches_empties():
    env = GomokuEnv(size=5)
    env.reset()
    env.step(12)
    mask = env.legal_mask()
    assert mask.sum() == 24
    assert not mask[12]
    assert np.array_equal(env.legal_actions(), np.flatnonzero(mask))
