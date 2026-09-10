from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

PARAMETER_NAMES = (
    "global_zero_shift_deg",
    "extra_instrument_fwhm_deg",
    "background_peak_ratio",
    "gaussian_noise_sigma_peak_ratio",
    "background_bernstein_0",
    "background_bernstein_1",
    "background_bernstein_2",
    "background_bernstein_3",
)


def _range_pair(value, name: str) -> tuple[float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 2 or not np.isfinite(values).all() or values[0] > values[1]:
        raise ValueError(f"{name} must contain a finite [min, max] pair")
    return values


@dataclass(frozen=True)
class ExperimentalObservationConfig:
    zero_shift_deg: tuple[float, float] = (-0.03, 0.03)
    extra_fwhm_deg: tuple[float, float] = (0.0, 0.05)
    background_peak_ratio: tuple[float, float] = (0.0, 0.01)
    gaussian_noise_sigma_peak_ratio: tuple[float, float] = (0.0008, 0.0020)

    @classmethod
    def from_mapping(cls, mapping) -> "ExperimentalObservationConfig":
        mapping = dict(mapping or {})
        config = cls(
            zero_shift_deg=_range_pair(
                mapping.get("zero_shift_deg", cls.zero_shift_deg), "zero_shift_deg"
            ),
            extra_fwhm_deg=_range_pair(
                mapping.get("extra_fwhm_deg", cls.extra_fwhm_deg), "extra_fwhm_deg"
            ),
            background_peak_ratio=_range_pair(
                mapping.get("background_peak_ratio", cls.background_peak_ratio),
                "background_peak_ratio",
            ),
            gaussian_noise_sigma_peak_ratio=_range_pair(
                mapping.get(
                    "gaussian_noise_sigma_peak_ratio",
                    cls.gaussian_noise_sigma_peak_ratio,
                ),
                "gaussian_noise_sigma_peak_ratio",
            ),
        )
        if config.extra_fwhm_deg[0] < 0.0:
            raise ValueError("extra_fwhm_deg cannot be negative")
        if config.background_peak_ratio[0] < 0.0:
            raise ValueError("background_peak_ratio cannot be negative")
        if config.gaussian_noise_sigma_peak_ratio[0] < 0.0:
            raise ValueError("gaussian_noise_sigma_peak_ratio cannot be negative")
        return config


class ExperimentalObservationModel:
    """One shared measurement process applied after phase-profile mixing."""

    parameter_names = PARAMETER_NAMES

    def __init__(self, axis_two_theta: np.ndarray, config=None):
        self.axis = np.asarray(axis_two_theta, dtype=np.float64)
        if self.axis.ndim != 1 or self.axis.size < 2:
            raise ValueError("axis_two_theta must be a one-dimensional grid")
        differences = np.diff(self.axis)
        if not np.all(differences > 0.0):
            raise ValueError("axis_two_theta must be strictly increasing")
        self.axis_step = float(np.median(differences))
        # The stored grid is float32, so nominal 0.02-degree steps carry a few
        # microdegrees of representation jitter.
        if not np.allclose(differences, self.axis_step, rtol=1.0e-4, atol=5.0e-6):
            raise ValueError("experimental observation currently requires a uniform grid")
        self.config = ExperimentalObservationConfig.from_mapping(config)

    @staticmethod
    def _sample_range(rng, bounds: tuple[float, float]) -> float:
        if bounds[0] == bounds[1]:
            return float(bounds[0])
        return float(rng.uniform(bounds[0], bounds[1]))

    def sample_parameters(self, rng) -> np.ndarray:
        background_anchors = np.asarray(rng.uniform(0.0, 1.0, size=4), dtype=np.float32)
        return np.asarray(
            [
                self._sample_range(rng, self.config.zero_shift_deg),
                self._sample_range(rng, self.config.extra_fwhm_deg),
                self._sample_range(rng, self.config.background_peak_ratio),
                self._sample_range(rng, self.config.gaussian_noise_sigma_peak_ratio),
                *background_anchors.tolist(),
            ],
            dtype=np.float32,
        )

    def _transform_component(
        self, component: np.ndarray, zero_shift_deg: float, extra_fwhm_deg: float
    ) -> np.ndarray:
        shifted = np.interp(
            self.axis - float(zero_shift_deg),
            self.axis,
            np.asarray(component, dtype=np.float64),
            left=0.0,
            right=0.0,
        )
        sigma_bins = float(extra_fwhm_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0))) / self.axis_step
        if sigma_bins > 1.0e-6:
            shifted = gaussian_filter1d(shifted, sigma=sigma_bins, mode="constant", truncate=4.0)
        return np.clip(shifted, 0.0, None)

    def _background(self, anchors: np.ndarray) -> np.ndarray:
        t = np.linspace(0.0, 1.0, self.axis.shape[0], dtype=np.float64)
        one_minus_t = 1.0 - t
        values = (
            anchors[0] * one_minus_t**3
            + 3.0 * anchors[1] * one_minus_t**2 * t
            + 3.0 * anchors[2] * one_minus_t * t**2
            + anchors[3] * t**3
        )
        return values / max(float(values.max()), 1.0e-12)

    @staticmethod
    def _normalize(pattern: np.ndarray) -> np.ndarray:
        pattern = np.clip(np.asarray(pattern, dtype=np.float64), 0.0, None)
        maximum = float(pattern.max())
        if maximum <= 0.0:
            return np.zeros(pattern.shape, dtype=np.float32)
        return (pattern / maximum).astype(np.float32)

    def build(
        self,
        phase_patterns: np.ndarray,
        phase_weights: np.ndarray,
        max_components: int,
        rng,
        return_common_scale: bool = False,
    ):
        phase_patterns = np.asarray(phase_patterns, dtype=np.float32)
        phase_weights = np.asarray(phase_weights, dtype=np.float32)
        if phase_patterns.ndim != 2 or phase_patterns.shape[1] != self.axis.shape[0]:
            raise ValueError("phase_patterns must have shape [K, L]")
        if phase_weights.shape != (phase_patterns.shape[0],):
            raise ValueError("phase_weights must align with phase_patterns")
        if phase_patterns.shape[0] < 1 or phase_patterns.shape[0] > int(max_components):
            raise ValueError("unsupported number of phase components")

        parameters = self.sample_parameters(rng)
        zero_shift_deg = float(parameters[0])
        extra_fwhm_deg = float(parameters[1])
        transformed = np.stack(
            [
                self._transform_component(pattern, zero_shift_deg, extra_fwhm_deg) * float(weight)
                for pattern, weight in zip(phase_patterns, phase_weights)
            ],
            axis=0,
        )
        structural_mixture = transformed.sum(axis=0)
        structural_peak = max(float(structural_mixture.max()), 1.0e-12)
        background = self._background(parameters[4:8]) * float(parameters[2]) * structural_peak
        noise = np.asarray(
            rng.normal(
                0.0,
                float(parameters[3]) * structural_peak,
                size=self.axis.shape[0],
            ),
            dtype=np.float64,
        )
        observed_raw = np.clip(structural_mixture + background + noise, 0.0, None)

        residuals = np.zeros((int(max_components) + 1, self.axis.shape[0]), dtype=np.float32)
        common_residuals = np.zeros_like(residuals)
        observation_scale = max(float(observed_raw.max()), 1.0e-12)
        explained = np.zeros(self.axis.shape[0], dtype=np.float64)
        for step in range(phase_patterns.shape[0] + 1):
            raw_residual = np.clip(observed_raw - explained, 0.0, None)
            residuals[step] = self._normalize(raw_residual)
            common_residuals[step] = np.asarray(raw_residual / observation_scale, dtype=np.float32)
            if step < phase_patterns.shape[0]:
                explained += transformed[step]
        transformed = transformed.astype(np.float32)
        result = residuals[0], residuals, parameters, transformed
        if not return_common_scale:
            return result

        component_contributions = np.zeros(
            (int(max_components), self.axis.shape[0]), dtype=np.float32
        )
        component_contributions[: transformed.shape[0]] = transformed / observation_scale
        common_scale = {
            "observation_scale": np.asarray([observation_scale], dtype=np.float32),
            "residual_patterns_common": common_residuals,
            "component_contributions": component_contributions,
        }
        return (*result, common_scale)
