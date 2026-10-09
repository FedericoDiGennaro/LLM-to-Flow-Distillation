"""Minimal, provenance-aware adapter for the gated PepDFM checkpoint.

The upstream assets are not redistributed by this repository.  They are fetched
at an immutable Hugging Face revision after the user accepts the MOG-DFM license.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


PEPDFM_REPO_ID = "ChatterjeeLab/MOG-DFM"
PEPDFM_REVISION = "fe115ca03bb2dab2f5aac2f2fbaae56e65c876d6"
PEPDFM_CHECKPOINT_RELATIVE_PATH = (
    "ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt"
)
PEPDFM_CHECKPOINT_SHA256 = (
    "0b7040a7f28bc7a96fa99a809b001236d5215ec520aa42d8a5a704adcd9aa442"
)
PEPDFM_VOCABULARY_SIZE = 24
PEPDFM_TOKEN_BY_ID = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
)
PEPDFM_REQUIRED_PATHS = (
    Path("README.md"),
    Path("PepDFM_unconditional_sample.py"),
    Path(PEPDFM_CHECKPOINT_RELATIVE_PATH),
    Path("flow_matching/path/__init__.py"),
    Path("flow_matching/path/scheduler/__init__.py"),
    Path("flow_matching/solver/__init__.py"),
    Path("flow_matching/utils/__init__.py"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PepDfmAssets:
    """Resolved, locally cached upstream assets."""

    root: Path
    checkpoint: Path

    @classmethod
    def resolve(cls, root: Path) -> "PepDfmAssets":
        root = root.resolve()
        missing = [
            str(path) for path in PEPDFM_REQUIRED_PATHS if not (root / path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "PepDFM assets are incomplete; run domains.peptides.fetch_assets. "
                f"Missing: {', '.join(missing)}"
            )
        checkpoint = root / PEPDFM_CHECKPOINT_RELATIVE_PATH
        observed_hash = sha256_file(checkpoint)
        if observed_hash != PEPDFM_CHECKPOINT_SHA256:
            raise ValueError(
                "PepDFM checkpoint hash mismatch: "
                f"expected {PEPDFM_CHECKPOINT_SHA256}, observed {observed_hash}"
            )
        return cls(root=root, checkpoint=checkpoint)


@dataclass(frozen=True)
class PepDfmSamplingConfig:
    """Controls for a finite, deterministic unconditional sampling batch."""

    length: int = 8
    num_samples: int = 20
    num_steps: int = 100
    seed: int = 1701
    source_distribution: str = "uniform"
    device: str = "auto"

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


@dataclass(frozen=True)
class PepDfmSample:
    """One decoded sample, preserving invalid token output for auditing."""

    sample_index: int
    token_ids: tuple[int, ...]
    sequence: str | None
    is_valid: bool
    invalid_reason: str | None = None


@dataclass(frozen=True)
class PepDfmSamplingResult:
    """Sample batch plus immutable upstream and runtime provenance."""

    repository_id: str
    revision: str
    checkpoint_sha256: str
    torch_version: str
    resolved_device: str
    config: PepDfmSamplingConfig
    samples: tuple[PepDfmSample, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "torch_version": self.torch_version,
            "resolved_device": self.resolved_device,
            "config": asdict(self.config),
            "samples": [asdict(sample) for sample in self.samples],
        }


class GaussianFourierProjection(nn.Module):
    """Architecture-compatible time features used by the PepDFM checkpoint."""

    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = value[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dense(value)[...]


class Swish(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(value) * value


class PepDfmCnnModel(nn.Module):
    """Inference-only reproduction of the upstream ``CNNModel`` architecture."""

    def __init__(
        self,
        alphabet_size: int = PEPDFM_VOCABULARY_SIZE,
        embed_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        self.token_embedding = nn.Embedding(alphabet_size, embed_dim)
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.swish = Swish()
        self.linear = nn.Conv1d(embed_dim, hidden_dim, kernel_size=9, padding=4)
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=9,
                    dilation=4,
                    padding=16,
                ),
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=9,
                    dilation=16,
                    padding=64,
                ),
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=9,
                    dilation=64,
                    padding=256,
                ),
            ]
        )
        self.denses = nn.ModuleList([Dense(embed_dim, hidden_dim) for _ in range(5)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, hidden_dim) for _ in range(5)])
        self.final = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, alphabet_size, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        value = self.token_embedding(tokens)
        time_embedding = self.swish(self.time_embed(time))
        value = value.permute(0, 2, 1)
        value = self.swish(self.linear(value))
        for block, dense, normalization in zip(
            self.blocks,
            self.denses,
            self.norms,
        ):
            hidden = self.swish(
                block(normalization(value + dense(time_embedding)[:, :, None]))
            )
            value = hidden + value if hidden.shape == value.shape else hidden
        value = self.final(value).permute(0, 2, 1)
        return value - value.mean(dim=-1, keepdim=True)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def _activate_bundled_flow_matching(assets: PepDfmAssets):
    root = str(assets.root)
    if root not in sys.path:
        sys.path.insert(0, root)
    importlib.invalidate_caches()
    flow_matching = importlib.import_module("flow_matching")
    module_path = Path(flow_matching.__file__).resolve()
    if not module_path.is_relative_to(assets.root):
        raise RuntimeError(
            "flow_matching resolved outside the pinned PepDFM asset directory: "
            f"{module_path}"
        )
    path_module = importlib.import_module("flow_matching.path")
    scheduler_module = importlib.import_module("flow_matching.path.scheduler")
    solver_module = importlib.import_module("flow_matching.solver")
    utils_module = importlib.import_module("flow_matching.utils")
    return (
        path_module.MixtureDiscreteProbPath,
        scheduler_module.PolynomialConvexScheduler,
        solver_module.MixtureDiscreteEulerSolver,
        utils_module.ModelWrapper,
    )


def _decode_sample(index: int, token_ids: tuple[int, ...]) -> PepDfmSample:
    if len(token_ids) < 2:
        return PepDfmSample(index, token_ids, None, False, "missing boundary tokens")
    if token_ids[0] != 0 or token_ids[-1] != 2:
        return PepDfmSample(
            index,
            token_ids,
            None,
            False,
            "model did not preserve <cls>/<eos> boundary tokens",
        )
    residue_ids = token_ids[1:-1]
    invalid_ids = sorted(
        {token_id for token_id in residue_ids if token_id < 4 or token_id >= 24}
    )
    if invalid_ids:
        return PepDfmSample(
            index,
            token_ids,
            None,
            False,
            f"non-residue token IDs in sequence: {invalid_ids}",
        )
    sequence = "".join(PEPDFM_TOKEN_BY_ID[token_id] for token_id in residue_ids)
    return PepDfmSample(index, token_ids, sequence, True)


def sample_pepdfm(
    assets: PepDfmAssets,
    config: PepDfmSamplingConfig,
) -> PepDfmSamplingResult:
    """Load the pinned checkpoint and generate one unconditional sample batch."""
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
        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
            del extras
            return torch.softmax(self.model(x, t), dim=-1)

    path = mixture_path_class(scheduler=scheduler_class(n=2.0))
    solver = solver_class(
        model=ProbabilityModelWrapper(model),
        path=path,
        vocabulary_size=PEPDFM_VOCABULARY_SIZE,
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
            (config.num_samples, 1),
            dtype=torch.long,
            device=device,
        )
        ending = torch.full(
            (config.num_samples, 1),
            2,
            dtype=torch.long,
            device=device,
        )
        initial = torch.cat([beginning, initial, ending], dim=1)
        solution = solver.sample(
            x_init=initial,
            step_size=1.0 / config.num_steps,
            verbose=False,
            time_grid=torch.tensor([0.0, 1.0 - 1e-3], device=device),
        )

    rows = solution.detach().cpu().tolist()
    samples = tuple(
        _decode_sample(index, tuple(int(token_id) for token_id in row))
        for index, row in enumerate(rows)
    )
    return PepDfmSamplingResult(
        repository_id=PEPDFM_REPO_ID,
        revision=PEPDFM_REVISION,
        checkpoint_sha256=PEPDFM_CHECKPOINT_SHA256,
        torch_version=torch.__version__,
        resolved_device=str(device),
        config=config,
        samples=samples,
    )
