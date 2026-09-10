"""Full-library Greedy/Beam; each path owns a history and a recomputed residual."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class Prediction:
    phase_ids: list[int]
    log_probability: float


@dataclass
class _State:
    owner: int
    history: list[int]
    query: np.ndarray
    score: float = 0.0


class Search:
    """Full-library search over max-normalized observations.

    ``network`` is an evaluation-mode STOP-stage PhaseMatcher; ``library`` is a
    SpectrumLibrary in checkpoint row order. Each input returns ranked Prediction
    objects. Decomposition uses sorted IDs, independently of selection order.
    """

    def __init__(self, network, library, *, decomposition_batch_size=64):
        self.network, self.library = network, library
        self.model = network.model
        self.device = next(network.parameters()).device
        self.limit = network.max_components
        self.chunk_size = int(decomposition_batch_size)
        if len(library) != network.num_phases or self.chunk_size < 1:
            raise ValueError("Invalid library size or decomposition batch size")

    def _validate_histories(self, histories, size):
        if len(histories) != size:
            raise ValueError("Provide one history per observation")
        for history in histories:
            if len(history) > self.limit or len(set(history)) != len(history):
                raise ValueError("History exceeds the phase limit or contains duplicates")
            if any(
                not isinstance(p, (int, np.integer)) or not 0 <= p < len(self.library)
                for p in history
            ):
                raise ValueError("History IDs must be integers in 0..N-1")

    @torch.no_grad()
    def decompose(self, observations, histories):
        """Return common-scale contributions/residuals in sorted phase-ID order.

        Empty histories return zero contributions and the unchanged observation.
        Results always have Kmax slots; valid_mask identifies the selected slots.
        """
        x = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError("Observations must have exactly one intensity channel")
            x = x[:, 0]
        if x.ndim != 2 or x.shape[1] != self.library.points:
            raise ValueError("Observations must be [B,L] on the reference grid")
        self._validate_histories(histories, len(x))
        ids = torch.full((len(x), self.limit), -1, dtype=torch.long, device=self.device)
        refs = x.new_zeros(len(x), self.limit, x.shape[-1])
        for row, history in enumerate(histories):
            ordered = sorted(history)
            if ordered:
                ids[row, : len(ordered)] = torch.as_tensor(ordered, device=self.device)
                refs[row, : len(ordered)] = torch.as_tensor(
                    self.library.phase_patterns(ordered), device=self.device
                )
        valid = ids >= 0
        result = dict(
            phase_ids=ids,
            valid_mask=valid,
            contributions=torch.zeros_like(refs),
            remainder=x.clone(),
            alignment_parameters=x.new_zeros(len(x), self.limit, 5),
        )
        rows = valid.any(dim=1).nonzero().flatten()
        for chunk in rows.split(self.chunk_size):
            if chunk.numel():
                out = self.network.spectral_decomposition(x[chunk], refs[chunk], valid[chunk])
                for name in ("contributions", "remainder", "alignment_parameters"):
                    result[name][chunk] = out[name]
        return result

    def _queries(self, observations, histories):
        residual = self.decompose(observations, histories)["remainder"].cpu().numpy()
        for row, history in enumerate(histories):
            if history:
                maximum = residual[row].max()
                residual[row] = residual[row] / max(maximum, 1e-6) if maximum > 1e-8 else 0.0
        return residual

    def _scores(self, states, library_embeddings, observation_tokens, use_stop):
        columns = len(self.library) + int(use_stop)
        scores = torch.empty(len(states), columns, device=self.device)
        # Group by history length: padding with BOS would change the last readout.
        for length in sorted({len(s.history) for s in states}):
            rows = [i for i, s in enumerate(states) if len(s.history) == length]
            subset = [states[i] for i in rows]
            query = torch.as_tensor(np.stack([s.query for s in subset]), device=self.device)
            history = torch.tensor(
                [[self.network.bos_id, *s.history] for s in subset],
                dtype=torch.long,
                device=self.device,
            )
            tokens = self.model.encode_patterns(query[:, None])
            phase = self.model.phase_logits(
                tokens,
                history,
                library_embeddings,
                self.network.bos_id,
                candidate_embeddings_normalized=True,
            )
            for row, state in enumerate(subset):
                phase[row, state.history] = -torch.inf
            if use_stop:
                owners = torch.tensor([s.owner for s in subset], device=self.device)
                stop = self.model.stop_logit(
                    observation_tokens[owners], history, self.network.bos_id
                )
                if length == 0:
                    stop.fill_(-torch.inf)
                phase = torch.cat([phase, stop[:, None]], dim=1)
            scores[rows] = phase
        return scores

    @torch.no_grad()
    def predict(self, mixtures, *, strategy="greedy", beam_size=10, histories=None, counts=None):
        """Return ranked paths per sample; counts is only for phase-stage validation.

        Beam keeps up to W unfinished AND W completed paths. Equivalent unordered
        sets are not deduplicated, matching the evaluation protocol.
        """
        if self.network.training:
            raise ValueError("Call network.eval() before search")
        if strategy not in {"greedy", "beam"} or int(beam_size) < 1:
            raise ValueError("Use greedy or beam with positive beam_size")
        width = 1 if strategy == "greedy" else int(beam_size)
        x = torch.as_tensor(mixtures, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x[:, None]
        if x.ndim != 3 or x.shape[1:] != (1, self.library.points):
            raise ValueError("Input must be [B,1,L] on the reference grid")
        if not torch.isfinite(x).all() or (x < 0).any():
            raise ValueError("Spectra must be finite and nonnegative")
        use_stop = counts is None
        if use_stop and not self.network.with_stop:
            raise ValueError("Automatic prediction requires a STOP-stage checkpoint")
        limits = [self.limit] * len(x) if use_stop else torch.as_tensor(counts).tolist()
        if len(limits) != len(x) or any(int(k) != k or not 1 <= k <= self.limit for k in limits):
            raise ValueError("Invalid fixed phase counts")
        histories = [[] for _ in x] if histories is None else [list(h) for h in histories]
        self._validate_histories(histories, len(x))
        if any(len(h) > k for h, k in zip(histories, limits)):
            raise ValueError("Initial history exceeds the requested count")
        observations = x[:, 0].cpu().numpy()
        queries = self._queries(observations, histories)
        active = [[_State(i, h, q)] for i, (h, q) in enumerate(zip(histories, queries))]
        finished = [[] for _ in x]
        embeddings = F.normalize(self.model.library_embeddings(), dim=1)
        observation_tokens = self.model.encode_patterns(x) if use_stop else None

        for _ in range(self.limit + 1):
            states = []
            for row, paths in enumerate(active):
                for state in paths:
                    if len(state.history) >= limits[row]:
                        finished[row].append(state)
                    else:
                        states.append(state)
            if not states:
                break
            logits = self._scores(states, embeddings, observation_tokens, use_stop)
            probabilities = F.log_softmax(logits, dim=1)
            if strategy == "greedy":
                # argmax chooses a phase on a phase/STOP tie (STOP is last).
                actions = logits.argmax(dim=1, keepdim=True)
                values = probabilities.gather(1, actions)
            else:
                values, actions = probabilities.topk(min(width, probabilities.shape[1]), dim=1)
            proposed = []
            for state, row_values, row_actions in zip(states, values.tolist(), actions.tolist()):
                for score, action in zip(row_values, row_actions):
                    if not np.isfinite(score):
                        continue
                    total = state.score + score
                    if use_stop and action == len(self.library):
                        finished[state.owner].append(
                            _State(state.owner, state.history, state.query, total)
                        )
                    else:
                        proposed.append(
                            _State(state.owner, [*state.history, action], state.query, total)
                        )
            # Recompute from the observation and ALL selected references, not
            # from the preceding residual. Chunking bounds GPU memory use.
            if proposed:
                queries = self._queries(
                    observations[[s.owner for s in proposed]], [s.history for s in proposed]
                )
                for state, query in zip(proposed, queries):
                    state.query = query
            active = [[] for _ in x]
            for state in proposed:
                active[state.owner].append(state)
            active = [
                sorted(paths, key=lambda s: s.score, reverse=True)[:width] for paths in active
            ]
            finished = [
                sorted(paths, key=lambda s: s.score, reverse=True)[:width] for paths in finished
            ]
        return [
            [
                Prediction(s.history, s.score)
                for s in sorted(paths, key=lambda s: s.score, reverse=True)[:width]
            ]
            for paths in finished
        ]
