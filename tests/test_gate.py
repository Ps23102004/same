"""The accept/reject rule in search.py. Pure logic: no model, no images."""
from backend.app.retrieval import search as S


def searcher(tmp_path):
    return S.Searcher(tmp_path / "index")


def test_constants_match_the_documented_operating_point():
    assert (S.MIN_INLIERS, S.MIN_INLIER_FRACTION, S.STRONG_INLIERS) == (6, 0.45, 30)


def test_strong_inliers_accept_regardless_of_fraction(tmp_path):
    assert searcher(tmp_path).accept(inliers=30, matches=10_000)


def test_two_gate_rule_boundaries(tmp_path):
    s = searcher(tmp_path)
    assert s.accept(inliers=6, matches=13)       # 6/13 = 0.462 >= 0.45
    assert not s.accept(inliers=6, matches=14)   # 6/14 = 0.429 <  0.45
    assert not s.accept(inliers=5, matches=5)    # fraction 1.0 but below the floor
    assert not s.accept(inliers=29, matches=1000)  # below STRONG, fraction 0.029
