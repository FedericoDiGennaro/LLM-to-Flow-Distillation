"""Portable adapter for MOG-DFM guidance of the pinned PepDFM model.

The released MOG-DFM entry point contains author-specific paths and omits two
arguments consumed by the sampler.  This module keeps the released solver and
guidance implementation pinned in the accepted upstream asset bundle while
making all runtime controls and objective identities explicit.

Every objective passed here must be a *high-is-better* token-level score.  The
same interface can therefore host a fixed scientific predictor, an analytical
mechanism test, or a learned utility distilled from LLM preferences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor, nn

from domains.peptides.model import PEPDFM_CHECKPOINT_SHA256, PEPDFM_REPO_ID, PEPDFM_REVISION, PEPDFM_VOCABULARY_SIZE, PepDfmAssets, PepDfmCnnModel, PepDfmSample, _activate_bundled_flow_matching, _decode_sample, _resolve_device


@runtime_checkable
class PeptideTokenObjective(Protocol):
    """A named, high-is-better score over a batch of tokenized peptides."""

    @property
    def objective_name(self) -> str:
        """Stable identifier recorded with generated samples."""

    def __call__(self, tokens: Tensor) -> Tensor:
        """Return one finite scalar per sequence, shaped ``[batch]``."""


@dataclass(frozen=True)
class MogGuidanceConfig:
    """Finite sampling controls plus the released MOG-DFM hyperparameters."""

    length: int = 8
    num_samples: int = 20
    num_steps: int = 100
    seed: int = 1701
    source_distribution: str = "uniform"
    device: str = "auto"
    num_div: int = 64
    lambda_: float = 1.0
    beta: float = 1.0
    alpha_r: float = 0.5
    eta: float = 1.0
    Phi_init: float = math.radians(45.0)
    Phi_min: float = math.radians(15.0)
    Phi_max: float = math.radians(75.0)
    tau: float = 0.3
    is_peptide: bool = True

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.num_steps < 2:
            raise ValueError("num_steps must be at least two")
        if self.source_distribution not in {"uniform", "mask"}:
            raise ValueError("source_distribution must be 'uniform' or 'mask'")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.num_div <= 0:
            raise ValueError("num_div must be positive")
        if not math.isfinite(self.lambda_) or self.lambda_ < 0:
            raise ValueError("lambda_ must be finite and non-negative")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and non-negative")
        if not 0 <= self.alpha_r <= 1:
            raise ValueError("alpha_r must lie in [0, 1]")
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and non-negative")
        if not 0 < self.Phi_min <= self.Phi_init <= self.Phi_max <= math.pi:
            raise ValueError(
                "cone angles must satisfy 0 < Phi_min <= Phi_init <= Phi_max <= pi"
            )
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must lie in [0, 1]")
        if not self.is_peptide:
            raise ValueError("the PepDFM adapter requires is_peptide=True")


@dataclass(frozen=True)
class PepDfmGuidedSamplingResult:
    """Guided samples with enough provenance to reproduce the baseline."""

    repository_id: str
    revision: str
    checkpoint_sha256: str
    torch_version: str
    resolved_device: str
    guidance_algorithm: str
    objective_names: tuple[str, ...]
    importance: tuple[float, ...]
    config: MogGuidanceConfig
    samples: tuple[PepDfmSample, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "torch_version": self.torch_version,
            "resolved_device": self.resolved_device,
            "guidance_algorithm": self.guidance_algorithm,
            "objective_names": list(self.objective_names),
            "importance": list(self.importance),
            "config": asdict(self.config),
            "samples": [asdict(sample) for sample in self.samples],
        }


class LowHydrophobicFractionObjective(nn.Module):
    """Cheap mechanism test matching MOG-DFM's released heuristic.

    This is ``1 - hydrophobic-residue fraction`` for the upstream residue set.
    It is deliberately labelled a mechanism test, not a validated solubility
    endpoint.  Boundary tokens are excluded from the denominator.
    """

    objective_name = "mechanism_test.low_hydrophobic_fraction"

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "hydrophobic_token_ids",
            torch.tensor([5, 7, 4, 12, 20, 18, 22, 14], dtype=torch.long),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] < 3:
            raise ValueError("tokens must have shape [batch, length >= 3]")
        residues = tokens[:, 1:-1]
        hydrophobic = (
            residues.unsqueeze(-1) == self.hydrophobic_token_ids.to(tokens.device)
        ).any(dim=-1)
        return 1.0 - hydrophobic.float().mean(dim=1)


class _CheckedObjective(nn.Module):
    """Fail early if an objective violates the sampler's score contract."""

    def __init__(self, objective: PeptideTokenObjective) -> None:
        super().__init__()
        self.objective = objective

    def forward(self, tokens: Tensor) -> Tensor:
        values = self.objective(tokens)
        if values.shape != (tokens.shape[0],):
            raise ValueError(
                f"objective {self.objective.objective_name!r} returned shape "
                f"{tuple(values.shape)}; expected {(tokens.shape[0],)}"
            )
        if not torch.isfinite(values).all():
            raise ValueError(
                f"objective {self.objective.objective_name!r} returned "
                "non-finite values"
            )
        return values


def _validate_objectives(
    objectives: Sequence[PeptideTokenObjective],
    importance: Sequence[float] | None,
) -> tuple[tuple[PeptideTokenObjective, ...], tuple[float, ...]]:
    resolved = tuple(objectives)
    if not resolved:
        raise ValueError("at least one guidance objective is required")
    names = [objective.objective_name.strip() for objective in resolved]
    if any(not name for name in names):
        raise ValueError("every guidance objective must have a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError("guidance objective names must be unique")
    weights = (
        tuple(1.0 for _ in resolved)
        if importance is None
        else tuple(float(value) for value in importance)
    )
    if len(weights) != len(resolved):
        raise ValueError("importance must contain one value per objective")
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("importance values must be finite and positive")
    return resolved, weights


def sample_pepdfm_with_mog_guidance(
    assets: PepDfmAssets,
    config: MogGuidanceConfig,
    objectives: Sequence[PeptideTokenObjective],
    importance: Sequence[float] | None = None,
) -> PepDfmGuidedSamplingResult:
    """Sample PepDFM through the released multi-objective guidance solver."""
    objectives, weights = _validate_objectives(objectives, importance)
    device = _resolve_device(config.device)
    (
        mixture_path_class,
        scheduler_class,
        solver_class,
        model_wrapper_class,
    ) = _activate_bundled_flow_matching(assets)

    model = PepDfmCnnModel().to(device)
    state_dict = torch.load(
        assets.checkpoint,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    class ProbabilityModelWrapper(model_wrapper_class):
        def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
            del extras
            return torch.softmax(self.model(x, t), dim=-1)

    path = mixture_path_class(scheduler=scheduler_class(n=2.0))
    solver = solver_class(
        model=ProbabilityModelWrapper(model),
        path=path,
        vocabulary_size=PEPDFM_VOCABULARY_SIZE,
    )
    score_models = tuple(
        _CheckedObjective(objective).to(device) for objective in objectives
    )
    random_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=random_devices), torch.inference_mode():
        torch.manual_seed(config.seed)
        if config.source_distribution == "uniform":
            initial = torch.randint(
                low=4,
                high=PEPDFM_VOCABULARY_SIZE,
                size=(config.num_samples, config.length),
                device=device,
            )
        else:
            initial = torch.full(
                (config.num_samples, config.length),
                3,
                dtype=torch.long,
                device=device,
            )
        beginning = torch.zeros(
            (config.num_samples, 1), dtype=torch.long, device=device
        )
        ending = torch.full((config.num_samples, 1), 2, dtype=torch.long, device=device)
        initial = torch.cat([beginning, initial, ending], dim=1)
        solution = solver.multi_guidance_sample(
            args=config,
            x_init=initial,
            step_size=1.0 / config.num_steps,
            verbose=False,
            time_grid=torch.tensor([0.0, 1.0 - 1e-3], device=device),
            score_models=list(score_models),
            importance=list(weights),
        )

    rows = solution.detach().cpu().tolist()
    samples = tuple(
        _decode_sample(index, tuple(int(token_id) for token_id in row))
        for index, row in enumerate(rows)
    )
    return PepDfmGuidedSamplingResult(
        repository_id=PEPDFM_REPO_ID,
        revision=PEPDFM_REVISION,
        checkpoint_sha256=PEPDFM_CHECKPOINT_SHA256,
        torch_version=torch.__version__,
        resolved_device=str(device),
        guidance_algorithm="MOG-DFM released adaptive-hypercone guidance",
        objective_names=tuple(objective.objective_name for objective in objectives),
        importance=weights,
        config=config,
        samples=samples,
    )
