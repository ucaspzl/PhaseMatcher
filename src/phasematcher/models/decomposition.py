from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(channels), int(maximum)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, padding=3, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 5, padding=2, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv1d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class _PyramidEncoder(nn.Module):
    def __init__(self, base_channels: int, levels: int):
        super().__init__()
        self.channels = [int(base_channels) * (2**level) for level in range(int(levels))]
        self.stem = _ConvBlock(2, self.channels[0])
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(source, target, 5, stride=2, padding=2, bias=False),
                    nn.GroupNorm(_group_count(target), target),
                    nn.GELU(),
                    _ConvBlock(target, target),
                )
                for source, target in zip(self.channels[:-1], self.channels[1:])
            ]
        )

    def forward(self, spectra: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = [self.stem(spectra)]
        for block in self.down:
            features.append(block(features[-1]))
        return tuple(features)


class _SetAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(int(channels), int(heads), batch_first=True)
        self.norm_attention = nn.LayerNorm(int(channels))
        self.feed_forward = nn.Sequential(
            nn.Linear(int(channels), 2 * int(channels)),
            nn.GELU(),
            nn.Linear(2 * int(channels), int(channels)),
        )
        self.norm_feed_forward = nn.LayerNorm(int(channels))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        safe_valid = valid.clone()
        empty_rows = ~safe_valid.any(dim=1)
        safe_valid[empty_rows, 0] = True
        safe_tokens = tokens * valid[:, :, None].to(tokens.dtype)
        attended, _ = self.attention(
            safe_tokens,
            safe_tokens,
            safe_tokens,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        tokens = self.norm_attention(safe_tokens + attended)
        tokens = self.norm_feed_forward(tokens + self.feed_forward(tokens))
        return tokens * valid[:, :, None].to(tokens.dtype)


class _PhaseDecoder(nn.Module):
    def __init__(self, channels: list[int], token_channels: int, initial_bias: float):
        super().__init__()
        self.bottleneck = _ConvBlock(4 * channels[-1], channels[-1])
        self.token_projections = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        previous = channels[-1]
        for target in reversed(channels[:-1]):
            self.token_projections.append(nn.Linear(token_channels, target))
            self.up_blocks.append(_ConvBlock(previous + 4 * target, target))
            previous = target
        self.output = nn.Conv1d(channels[0], 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, float(initial_bias))

    def forward(
        self,
        observation: tuple[torch.Tensor, ...],
        reference: tuple[torch.Tensor, ...],
        set_features: tuple[torch.Tensor, ...],
        phase_token: torch.Tensor,
    ) -> torch.Tensor:
        token = phase_token[..., None].expand(-1, -1, observation[-1].shape[-1])
        x = self.bottleneck(
            torch.cat([observation[-1], reference[-1], set_features[-1], token], dim=1)
        )
        for level, (projection, block) in enumerate(
            zip(self.token_projections, self.up_blocks), start=2
        ):
            observation_level = observation[-level]
            reference_level = reference[-level]
            set_level = set_features[-level]
            x = F.interpolate(
                x, size=observation_level.shape[-1], mode="linear", align_corners=False
            )
            projected = projection(phase_token)[..., None].expand(
                -1, -1, observation_level.shape[-1]
            )
            x = block(
                torch.cat(
                    [x, observation_level, reference_level, set_level, projected],
                    dim=1,
                )
            )
        return self.output(x).squeeze(1)


class _RemainderDecoder(nn.Module):
    def __init__(self, channels: list[int], token_channels: int, initial_bias: float):
        super().__init__()
        self.bottleneck = _ConvBlock(3 * channels[-1], channels[-1])
        self.token_projections = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        previous = channels[-1]
        for target in reversed(channels[:-1]):
            self.token_projections.append(nn.Linear(token_channels, target))
            self.up_blocks.append(_ConvBlock(previous + 3 * target, target))
            previous = target
        self.output = nn.Conv1d(channels[0], 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, float(initial_bias))

    def forward(
        self,
        observation: tuple[torch.Tensor, ...],
        set_features: tuple[torch.Tensor, ...],
        set_token: torch.Tensor,
    ) -> torch.Tensor:
        token = set_token[..., None].expand(-1, -1, observation[-1].shape[-1])
        x = self.bottleneck(torch.cat([observation[-1], set_features[-1], token], dim=1))
        for level, (projection, block) in enumerate(
            zip(self.token_projections, self.up_blocks), start=2
        ):
            observation_level = observation[-level]
            set_level = set_features[-level]
            x = F.interpolate(
                x, size=observation_level.shape[-1], mode="linear", align_corners=False
            )
            projected = projection(set_token)[..., None].expand(-1, -1, observation_level.shape[-1])
            x = block(torch.cat([x, observation_level, set_level, projected], dim=1))
        return self.output(x).squeeze(1)


class PhysicsGuidedSpectralDecomposition(nn.Module):
    """Set-conditioned conservative decomposition of a PXRD observation."""

    def __init__(
        self,
        base_channels: int = 24,
        levels: int = 4,
        attention_heads: int = 4,
        attention_layers: int = 2,
        max_shift_bins: float = 10.0,
        max_strain: float = 0.005,
        support_radius_bins: int = 15,
        phase_logit_bias: float = -0.5,
        remainder_logit_bias: float = 0.5,
        two_theta_min_deg: float = 10.0,
        two_theta_max_deg: float = 80.0,
        alignment_regions: int = 8,
        correlation_logit_scale: float = 20.0,
    ):
        super().__init__()
        if int(levels) < 2:
            raise ValueError("levels must be at least two")
        self.base_channels = int(base_channels)
        self.levels = int(levels)
        self.max_shift_bins = float(max_shift_bins)
        self.max_strain = float(max_strain)
        self.support_radius_bins = int(support_radius_bins)
        self.two_theta_min_deg = float(two_theta_min_deg)
        self.two_theta_max_deg = float(two_theta_max_deg)
        self.alignment_regions = int(alignment_regions)
        if self.max_shift_bins < 0.0:
            raise ValueError("max_shift_bins must be nonnegative")
        if not math.isclose(self.max_shift_bins, round(self.max_shift_bins), abs_tol=1.0e-6):
            raise ValueError(
                "max_shift_bins must be an integer-valued number for the "
                "explicit correlation volume"
            )
        if self.max_strain < 0.0:
            raise ValueError("max_strain must be nonnegative")
        if self.support_radius_bins < 0:
            raise ValueError("support_radius_bins must be nonnegative")
        if not self.two_theta_min_deg < self.two_theta_max_deg:
            raise ValueError("two_theta_min_deg must be smaller than two_theta_max_deg")
        if self.alignment_regions < 1:
            raise ValueError("alignment_regions must be positive")
        if float(correlation_logit_scale) <= 0.0:
            raise ValueError("correlation_logit_scale must be positive")

        self.correlation_radius_bins = int(round(self.max_shift_bins))
        self.register_buffer(
            "shift_candidate_bins",
            torch.arange(
                -self.correlation_radius_bins,
                self.correlation_radius_bins + 1,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.observation_encoder = _PyramidEncoder(self.base_channels, self.levels)
        self.reference_encoder = _PyramidEncoder(self.base_channels, self.levels)
        feature_channels = self.observation_encoder.channels
        token_channels = feature_channels[-1]

        correlation_channels = self.alignment_regions * int(self.shift_candidate_bins.numel())
        self.global_shift_head = nn.Sequential(
            nn.Linear(int(self.shift_candidate_bins.numel()), token_channels),
            nn.GELU(),
            nn.Linear(token_channels, int(self.shift_candidate_bins.numel())),
        )
        nn.init.zeros_(self.global_shift_head[-1].weight)
        nn.init.zeros_(self.global_shift_head[-1].bias)
        self.correlation_logit_scale = nn.Parameter(
            torch.tensor(math.log(float(correlation_logit_scale)))
        )

        self.alignment_head = nn.Sequential(
            nn.Linear(2 * token_channels + correlation_channels, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, 4),
        )
        alignment_output = self.alignment_head[-1]
        nn.init.zeros_(alignment_output.weight)
        with torch.no_grad():
            alignment_output.bias.copy_(torch.tensor([0.0, 5.0, -2.0, -4.0]))

        self.set_attention = nn.ModuleList(
            [
                _SetAttentionBlock(token_channels, attention_heads)
                for _ in range(int(attention_layers))
            ]
        )
        self.phase_decoder = _PhaseDecoder(
            feature_channels, token_channels, initial_bias=phase_logit_bias
        )
        self.remainder_decoder = _RemainderDecoder(
            feature_channels, token_channels, initial_bias=remainder_logit_bias
        )
        self.support_log_strength = nn.Parameter(torch.tensor(-0.5))

    @staticmethod
    def _validate_inputs(
        original_observation: torch.Tensor,
        references: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if original_observation.ndim != 2:
            raise ValueError("original_observation must have shape [B, L]")
        if references.ndim != 3:
            raise ValueError("references must have shape [B, K, L]")
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape [B, K]")
        batch, points = original_observation.shape
        if references.shape[0] != batch or references.shape[2] != points:
            raise ValueError("references must align with original_observation")
        if tuple(valid_mask.shape) != tuple(references.shape[:2]):
            raise ValueError("valid_mask must align with references")
        if not torch.isfinite(original_observation).all():
            raise ValueError("original_observation must be finite")
        if (original_observation < 0.0).any():
            raise ValueError("original_observation must be nonnegative")
        if not torch.isfinite(references).all():
            raise ValueError("references must be finite")
        valid = valid_mask.to(device=references.device, dtype=torch.bool)
        if valid.any():
            valid_references = references[valid]
            if (valid_references < 0.0).any():
                raise ValueError("valid references must be nonnegative")
            if (valid_references.amax(dim=-1) <= 0.0).any():
                raise ValueError("every valid reference must contain positive intensity")

    @staticmethod
    def _spectrum_channels(spectra: torch.Tensor) -> torch.Tensor:
        spectra = spectra.clamp_min(0.0)
        log_spectra = torch.log1p(9.0 * spectra) / math.log(10.0)
        return torch.stack([spectra, log_spectra], dim=1)

    @staticmethod
    def _shift_signal(signal: torch.Tensor, shift_bins: int) -> torch.Tensor:
        """Translate a spectrum without wraparound; positive shifts move right."""
        shift_bins = int(shift_bins)
        if shift_bins == 0:
            return signal
        if abs(shift_bins) >= signal.shape[-1]:
            return torch.zeros_like(signal)
        if shift_bins > 0:
            return F.pad(signal[..., :-shift_bins], (shift_bins, 0))
        magnitude = -shift_bins
        return F.pad(signal[..., magnitude:], (0, magnitude))

    def _correlation_volume(
        self, observation: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        """Local cosine correlations over explicit integer shift candidates."""
        points = int(observation.shape[-1])
        observation_signal = torch.log1p(9.0 * observation.clamp_min(0.0))
        reference_signal = torch.log1p(9.0 * references.clamp_min(0.0))
        shifted_references = torch.stack(
            [
                self._shift_signal(reference_signal, shift)
                for shift in range(
                    -self.correlation_radius_bins,
                    self.correlation_radius_bins + 1,
                )
            ],
            dim=2,
        )
        observation_signal = observation_signal[:, None, None, :]
        region_correlations = []
        for region in range(self.alignment_regions):
            start = region * points // self.alignment_regions
            end = (region + 1) * points // self.alignment_regions
            if end <= start:
                region_correlations.append(
                    shifted_references.new_zeros(shifted_references.shape[:3])
                )
                continue
            observation_region = observation_signal[..., start:end]
            reference_region = shifted_references[..., start:end]
            numerator = (observation_region * reference_region).sum(dim=-1)
            denominator = (
                observation_region.square().sum(dim=-1).sqrt()
                * reference_region.square().sum(dim=-1).sqrt()
            )
            region_correlations.append(
                torch.where(
                    denominator > 1.0e-8,
                    numerator / denominator.clamp_min(1.0e-8),
                    torch.zeros_like(numerator),
                )
            )
        return torch.stack(region_correlations, dim=2)

    def _global_shift_from_correlation(
        self, correlation_volume: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        valid_values = valid.to(correlation_volume.dtype)
        pooled = (correlation_volume * valid_values[:, :, None, None]).sum(dim=(1, 2))
        pooled = pooled / (
            valid_values.sum(dim=1).clamp_min(1.0)[:, None] * float(self.alignment_regions)
        )
        logits = self.correlation_logit_scale.exp().clamp_max(
            100.0
        ) * pooled + self.global_shift_head(pooled)
        probabilities = torch.softmax(logits, dim=-1)
        shift_bins = (
            probabilities
            * self.shift_candidate_bins.to(device=probabilities.device, dtype=probabilities.dtype)[
                None, :
            ]
        ).sum(dim=-1)
        return shift_bins * valid.any(dim=1).to(shift_bins.dtype)

    @staticmethod
    def _bragg_source_two_theta(
        target_two_theta: torch.Tensor,
        shift_degrees: torch.Tensor,
        strain: torch.Tensor,
    ) -> torch.Tensor:
        """Map observed coordinates to the clean reference via Bragg's law."""
        shifted_target = target_two_theta - shift_degrees[:, None]
        output_theta = torch.deg2rad(shifted_target / 2.0)
        inverse_argument = (1.0 + strain[:, None]) * torch.sin(output_theta)
        inverse_argument = inverse_argument.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
        return torch.rad2deg(2.0 * torch.asin(inverse_argument))

    @classmethod
    def _warp(
        cls,
        features: torch.Tensor,
        shift_degrees: torch.Tensor,
        strain: torch.Tensor,
        two_theta_min_deg: float = 10.0,
        two_theta_max_deg: float = 80.0,
    ) -> torch.Tensor:
        batch, _channels, points = features.shape
        coordinate_dtype = (
            torch.float32 if features.dtype in {torch.float16, torch.bfloat16} else features.dtype
        )
        target_two_theta = torch.linspace(
            float(two_theta_min_deg),
            float(two_theta_max_deg),
            points,
            device=features.device,
            dtype=coordinate_dtype,
        )[None, :].expand(batch, -1)
        source_two_theta = cls._bragg_source_two_theta(
            target_two_theta,
            shift_degrees.to(device=features.device, dtype=coordinate_dtype),
            strain.to(device=features.device, dtype=coordinate_dtype),
        )
        source = (
            2.0
            * (source_two_theta - float(two_theta_min_deg))
            / float(two_theta_max_deg - two_theta_min_deg)
            - 1.0
        )
        grid = features.new_zeros((batch, 1, points, 2))
        grid[:, 0, :, 0] = source.to(features.dtype)
        return F.grid_sample(
            features[:, :, None, :],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)

    @staticmethod
    def _blur_mixture(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        candidates = torch.stack(
            [
                features,
                F.avg_pool1d(features, 3, stride=1, padding=1),
                F.avg_pool1d(features, 7, stride=1, padding=3),
            ],
            dim=1,
        )
        return (candidates * weights[:, :, None, None]).sum(dim=1)

    def forward(
        self,
        original_observation: torch.Tensor,
        references: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(original_observation, references, valid_mask)
        batch, points = original_observation.shape
        reference_count = int(references.shape[1])
        valid = valid_mask.to(device=references.device, dtype=torch.bool)
        valid_values = valid.to(dtype=references.dtype)

        if reference_count == 0 or not valid.any():
            masks = original_observation.new_zeros((batch, reference_count + 1, points))
            masks[:, -1] = 1.0
            return {
                "contributions": original_observation.new_zeros((batch, reference_count, points)),
                "remainder": original_observation,
                "source_masks": masks,
                "phase_logits": original_observation.new_full(
                    (batch, reference_count, points), -torch.inf
                ),
                "remainder_logits": original_observation.new_zeros((batch, points)),
                "alignment_parameters": original_observation.new_zeros((batch, reference_count, 5)),
            }

        observation_pyramid = self.observation_encoder(
            self._spectrum_channels(original_observation)
        )
        safe_references = references * valid_values[:, :, None]
        correlation_volume = self._correlation_volume(original_observation, safe_references)
        correlation_volume = correlation_volume * valid_values[:, :, None, None]
        global_shift_bins = self._global_shift_from_correlation(correlation_volume, valid)
        axis_step_degrees = (self.two_theta_max_deg - self.two_theta_min_deg) / max(points - 1, 1)
        global_shift_degrees = global_shift_bins * axis_step_degrees

        flat_references = safe_references.reshape(batch * reference_count, points)
        reference_pyramid = self.reference_encoder(self._spectrum_channels(flat_references))

        observation_token = observation_pyramid[-1].mean(dim=-1)
        reference_token = reference_pyramid[-1].mean(dim=-1).reshape(batch, reference_count, -1)
        alignment_input = torch.cat(
            [
                observation_token[:, None].expand(-1, reference_count, -1),
                reference_token,
                correlation_volume.flatten(start_dim=2),
            ],
            dim=-1,
        )
        raw_alignment = self.alignment_head(alignment_input)
        strain = self.max_strain * torch.tanh(raw_alignment[..., 0])
        blur_weights = torch.softmax(raw_alignment[..., 1:4], dim=-1)
        flat_shift_degrees = global_shift_degrees[:, None].expand(-1, reference_count).reshape(-1)

        aligned_reference_pyramid = []
        for features in reference_pyramid:
            warped = self._warp(
                features,
                flat_shift_degrees,
                strain.reshape(-1),
                self.two_theta_min_deg,
                self.two_theta_max_deg,
            )
            blurred = self._blur_mixture(warped, blur_weights.reshape(batch * reference_count, 3))
            aligned_reference_pyramid.append(
                blurred.reshape(batch, reference_count, blurred.shape[1], blurred.shape[2])
                * valid_values[:, :, None, None]
            )

        phase_tokens = aligned_reference_pyramid[-1].mean(dim=-1)
        for block in self.set_attention:
            phase_tokens = block(phase_tokens, valid)
        denominator = valid_values.sum(dim=1).clamp_min(1.0)
        set_token = (phase_tokens * valid_values[:, :, None]).sum(dim=1) / denominator[:, None]
        set_pyramid = tuple(
            (features * valid_values[:, :, None, None]).sum(dim=1) / denominator[:, None, None]
            for features in aligned_reference_pyramid
        )

        flat_observation = tuple(
            feature[:, None]
            .expand(-1, reference_count, -1, -1)
            .reshape(batch * reference_count, feature.shape[1], feature.shape[2])
            for feature in observation_pyramid
        )
        flat_reference = tuple(
            feature.reshape(batch * reference_count, feature.shape[2], feature.shape[3])
            for feature in aligned_reference_pyramid
        )
        flat_set = tuple(
            feature[:, None]
            .expand(-1, reference_count, -1, -1)
            .reshape(batch * reference_count, feature.shape[1], feature.shape[2])
            for feature in set_pyramid
        )
        phase_logits = self.phase_decoder(
            flat_observation,
            flat_reference,
            flat_set,
            phase_tokens.reshape(batch * reference_count, -1),
        ).reshape(batch, reference_count, points)

        peak_threshold = safe_references.amax(dim=-1, keepdim=True) * 1.0e-3
        support = (safe_references > peak_threshold.clamp_min(1.0e-8)).to(references.dtype)
        if self.support_radius_bins:
            kernel = 2 * self.support_radius_bins + 1
            support = F.max_pool1d(
                support.reshape(batch * reference_count, 1, points),
                kernel,
                stride=1,
                padding=self.support_radius_bins,
            ).reshape(batch, reference_count, points)
        aligned_support = self._warp(
            support.reshape(batch * reference_count, 1, points),
            flat_shift_degrees,
            strain.reshape(-1),
            self.two_theta_min_deg,
            self.two_theta_max_deg,
        ).reshape(batch, reference_count, points)
        support_strength = F.softplus(self.support_log_strength)
        phase_logits = phase_logits + support_strength * torch.log(aligned_support.clamp_min(0.05))
        phase_logits = phase_logits.masked_fill(~valid[:, :, None], -torch.inf)

        remainder_logits = self.remainder_decoder(observation_pyramid, set_pyramid, set_token)
        source_logits = torch.cat([phase_logits, remainder_logits[:, None, :]], dim=1)
        source_masks = torch.softmax(source_logits, dim=1)
        contributions = original_observation[:, None, :] * source_masks[:, :-1]
        remainder = original_observation * source_masks[:, -1]
        shift_bins = global_shift_bins[:, None].expand(-1, reference_count)
        alignment_parameters = (
            torch.cat([shift_bins[..., None], strain[..., None], blur_weights], dim=-1)
            * valid_values[:, :, None]
        )

        return {
            "contributions": contributions,
            "remainder": remainder,
            "source_masks": source_masks,
            "phase_logits": phase_logits,
            "remainder_logits": remainder_logits,
            "alignment_parameters": alignment_parameters,
        }
