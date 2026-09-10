"""Sample-averaged set metrics, with explicit true-phase-count breakdowns."""

import numpy as np


class SetMetrics:
    names = ("exact_set_at_1", "phase_recall", "count_accuracy", "exact_set_at_10")

    def __init__(self, max_components=4):
        self.sums = np.zeros((max_components + 1, 5), dtype=np.float64)

    def update(self, predictions, targets):
        for paths, target in zip(predictions, targets):
            actual = set(int(p) for p in target if p >= 0)
            sets = [set(path.phase_ids) for path in paths]
            best = sets[0] if sets else set()
            scores = [
                1,
                best == actual,
                len(best & actual) / max(len(actual), 1),
                len(best) == len(actual),
                any(s == actual for s in sets[:10]),
            ]
            self.sums[0] += scores
            self.sums[len(actual)] += scores

    def result(self):
        def row(values):
            return {
                "samples": int(values[0]),
                **{
                    name: float(value / values[0]) if values[0] else None
                    for name, value in zip(self.names, values[1:])
                },
            }

        return {
            "overall": row(self.sums[0]),
            "by_phase_count": {str(k): row(values) for k, values in enumerate(self.sums[1:], 1)},
        }
