import pytest
import torch
from phasematcher.inference import Search
from phasematcher.models import PhaseMatcher


def test_search_stop_masks_and_cap(cfg, library, batch):
    model = PhaseMatcher(cfg.model, cfg.decomposition).eval()
    with torch.no_grad():
        model.model.stop_head.weight.zero_()
        model.model.stop_head.bias.fill_(1000)
    search = Search(model, library)
    for strategy in ("greedy", "beam"):
        result = search.predict(batch["mixture"], strategy=strategy, beam_size=3)
        assert all(len(paths[0].phase_ids) == 1 for paths in result)
        assert all(len(paths) <= (1 if strategy == "greedy" else 3) for paths in result)
    with torch.no_grad():
        model.model.stop_head.bias.fill_(-1000)
    result = search.predict(batch["mixture"])
    assert all(len(set(paths[0].phase_ids)) == 4 for paths in result)


def test_fixed_count_and_mixed_initial_histories(cfg, library, batch):
    model = PhaseMatcher(cfg.model, cfg.decomposition).eval()
    search = Search(model, library)
    result = search.predict(batch["mixture"], counts=batch["counts"])
    assert [len(p[0].phase_ids) for p in result] == [1, 2, 3, 4]
    histories = [[], [0], [1, 2], [0, 1, 2, 3]]
    for strategy in ("greedy", "beam"):
        together = search.predict(
            batch["mixture"], histories=histories, strategy=strategy, beam_size=2
        )
        for row, history in enumerate(histories):
            alone = search.predict(
                batch["mixture"][row : row + 1], histories=[history], strategy=strategy, beam_size=2
            )
            assert [p.phase_ids for p in together[row]] == [p.phase_ids for p in alone[0]]
            assert together[row][0].phase_ids[: len(history)] == history
    for invalid in ([[0, 0]], [[6]], [[0, 1, 2, 3, 4]], [[1.5]]):
        with pytest.raises(ValueError):
            search.predict(batch["mixture"][:1], histories=invalid)


def test_decomposition_sorted_sets_and_common_scale(cfg, library, batch):
    search = Search(PhaseMatcher(cfg.model, cfg.decomposition).eval(), library)
    x = batch["mixture"][:1]
    a = search.decompose(x, [[2, 0]])
    b = search.decompose(x, [[0, 2]])
    assert a["phase_ids"].tolist() == [[0, 2, -1, -1]]
    torch.testing.assert_close(a["contributions"], b["contributions"])
    torch.testing.assert_close(a["contributions"].sum(1) + a["remainder"], x[:, 0])
