import torch
import torch.nn.functional as F
from torch import nn

from .encoder import SpectrumTokenEncoder


class PhaseHistoryEmbedding(nn.Module):
    def __init__(self, d_model, max_positions):
        super().__init__()
        self.bos = nn.Parameter(torch.randn(d_model) * 0.02)
        self.position = nn.Parameter(torch.randn(1, max_positions, d_model) * 0.02)

    def forward(self, history_phase_ids, phase_table, bos_id):
        batch, steps = history_phase_ids.shape
        output = torch.zeros(
            (batch, steps, self.bos.numel()),
            device=history_phase_ids.device,
            dtype=phase_table.weight.dtype,
        )
        bos_mask = history_phase_ids == int(bos_id)
        phase_mask = ~bos_mask
        output[bos_mask] = self.bos
        if phase_mask.any():
            output[phase_mask] = phase_table(history_phase_ids[phase_mask])
        return output + self.position[:, :steps]


class CrossAttentionDecoder(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, layers, dropout):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(layers))

    def forward(self, history_phase, spectrum_embeddings):
        history_length = history_phase.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                (history_length, history_length), dtype=torch.bool, device=history_phase.device
            ),
            diagonal=1,
        )
        return self.decoder(tgt=history_phase, memory=spectrum_embeddings, tgt_mask=causal_mask)


class PhaseSelector(nn.Module):
    def __init__(self, d_model, temperature):
        super().__init__()
        self.query = nn.Linear(int(d_model), int(d_model))
        self.temperature = float(temperature)

    def forward(
        self, decoder_tokens, candidate_embeddings, *, candidate_embeddings_normalized=False
    ):
        phase_query = F.normalize(self.query(decoder_tokens[:, -1]), dim=1)
        candidates = (
            candidate_embeddings
            if candidate_embeddings_normalized
            else F.normalize(candidate_embeddings, dim=1)
        )
        return phase_query @ candidates.transpose(0, 1) / self.temperature


class PhaseRetriever(nn.Module):
    """Predict the next phase from the residual query and selected history."""

    def __init__(self, cfg):
        super().__init__()
        d_model = int(cfg.d_model)
        self.spectrum_encoder = SpectrumTokenEncoder(cfg.spectrum_encoder)
        self.phase_table = nn.Embedding(int(cfg.num_phases), d_model)
        nn.init.normal_(self.phase_table.weight, mean=0.0, std=d_model**-0.5)
        self.history = PhaseHistoryEmbedding(d_model, int(cfg.max_components) + 1)
        self.phase_decoder = CrossAttentionDecoder(
            d_model, cfg.nhead, cfg.ff_dim, cfg.decoder_layers, cfg.dropout
        )
        self.phase_selector = PhaseSelector(d_model, cfg.temperature)

    def encode_patterns(self, patterns):
        return self.spectrum_encoder(patterns)

    def candidate_embeddings(self, candidate_ids):
        return self.phase_table(candidate_ids)

    def library_embeddings(self):
        return self.phase_table.weight

    def phase_logits(
        self,
        residual_embeddings,
        history_phase_ids,
        candidate_embeddings,
        bos_id,
        *,
        candidate_embeddings_normalized=False,
    ):
        history_embeddings = self.history(history_phase_ids, self.phase_table, bos_id)
        decoder_tokens = self.phase_decoder(history_embeddings, residual_embeddings)
        return self.phase_selector(
            decoder_tokens,
            candidate_embeddings,
            candidate_embeddings_normalized=candidate_embeddings_normalized,
        )


class PhaseStopRetriever(PhaseRetriever):
    """Phase retriever plus a separate STOP decoder that shares phase history."""

    def __init__(self, cfg):
        super().__init__(cfg)
        d_model = int(cfg.d_model)
        stop_layers = int(cfg.get("stop_decoder_layers", cfg.decoder_layers))
        self.stop_decoder = CrossAttentionDecoder(
            d_model, cfg.nhead, cfg.ff_dim, stop_layers, cfg.dropout
        )
        self.stop_head = nn.Linear(d_model, 1)

    def stop_logit(self, mixture_tokens, history_phase_ids, bos_id):
        history_tokens = self.history(history_phase_ids, self.phase_table, bos_id)
        decoder_tokens = self.stop_decoder(history_tokens, mixture_tokens)
        return self.stop_head(decoder_tokens[:, -1]).squeeze(1)
