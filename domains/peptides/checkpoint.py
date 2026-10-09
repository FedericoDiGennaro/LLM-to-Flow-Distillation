"""Trainable PepDFM checkpoint, corruption loss, and sampling utilities."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import importlib
import io
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import torch
from domains.peptides.model import (
    PEPDFM_CHECKPOINT_SHA256,
    PEPDFM_TOKEN_BY_ID,
    PEPDFM_VOCABULARY_SIZE,
    PepDfmAssets,
    PepDfmCnnModel,
    PepDfmSample,
    _activate_bundled_flow_matching,
    _decode_sample,
    _resolve_device,
)

TOKEN_ID_BY_RESIDUE = {
    token: index for index, token in enumerate(PEPDFM_TOKEN_BY_ID) if index >= 4
}


@dataclass(frozen=True)
class LoadedPepDfmCheckpoint:
    """Provenance for an explicitly pinned, architecture-checked checkpoint."""

    path: str
    file_sha256: str
    state_dict_sha256: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def encode_peptide(sequence: str) -> tuple[int, ...]:
    """Encode a canonical peptide using PepDFM's released token vocabulary."""
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("peptide sequence must be non-empty")
    invalid = sorted(set(normalized).difference(TOKEN_ID_BY_RESIDUE))
    if invalid:
        raise ValueError(f"non-canonical peptide residues: {invalid}")
    return (0, *(TOKEN_ID_BY_RESIDUE[residue] for residue in normalized), 2)


def stratified_hash_split(
    sequences: Sequence[str], *, holdout_fraction: float, salt: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create a deterministic non-empty train/holdout split without RNG state."""
    unique = tuple(sorted(set(sequences)))
    if len(unique) < 2:
        raise ValueError("each class requires at least two unique sequences")
    if not math.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must lie strictly between zero and one")
    ordered = sorted(
        unique,
        key=lambda sequence: hashlib.sha256(
            f"{salt}:{sequence}".encode("utf-8")
        ).hexdigest(),
    )
    holdout_count = min(
        len(ordered) - 1, max(1, int(round(len(ordered) * holdout_fraction)))
    )
    holdout = tuple(ordered[:holdout_count])
    train = tuple(ordered[holdout_count:])
    return (train, holdout)


def sha256_state_dict(model: torch.nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and exact CPU bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class PepDfmCheckpoint:
    """A trainable copy of the pinned PepDFM model and its original loss."""

    def __init__(self, assets: PepDfmAssets, *, device: str = "auto") -> None:
        self.assets = assets
        self.device = _resolve_device(device)
        (
            self._mixture_path_class,
            self._scheduler_class,
            self._solver_class,
            self._model_wrapper_class,
        ) = _activate_bundled_flow_matching(assets)
        loss_module = importlib.import_module("flow_matching.loss")
        self._loss_class = loss_module.MixturePathGeneralizedKL
        self.path = self._mixture_path_class(scheduler=self._scheduler_class(n=2.0))
        self.model = PepDfmCnnModel().to(self.device)
        state_dict = torch.load(
            assets.checkpoint, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        if sha256_file(assets.checkpoint) != PEPDFM_CHECKPOINT_SHA256:
            raise ValueError("pinned PepDFM checkpoint changed after asset validation")

    def clone(self) -> "PepDfmCheckpoint":
        clone = PepDfmCheckpoint(self.assets, device=str(self.device))
        clone.model.load_state_dict(self.model.state_dict(), strict=True)
        clone.model.eval()
        return clone

    def load_checkpoint(
        self, path: Path, *, expected_sha256: str
    ) -> LoadedPepDfmCheckpoint:
        """Load an exact state-dict checkpoint without trusting pickle payloads.

        The file is read once into memory so the bytes that are hashed are also
        the bytes passed to ``torch.load``.  ``weights_only=True`` limits the
        accepted serialization surface, and keys, shapes, and dtypes must match
        the released PepDFM architecture before any state is installed.
        """
        normalized_hash = expected_sha256.strip().lower()
        if re.fullmatch("[0-9a-f]{64}", normalized_hash) is None:
            raise ValueError(
                "expected checkpoint SHA-256 must be 64 lowercase hex digits"
            )
        resolved = path.resolve(strict=True)
        payload_bytes = resolved.read_bytes()
        observed_hash = hashlib.sha256(payload_bytes).hexdigest()
        if observed_hash != normalized_hash:
            raise ValueError(
                f"parent checkpoint SHA-256 mismatch: expected {normalized_hash}, observed {observed_hash}"
            )
        payload = torch.load(
            io.BytesIO(payload_bytes), map_location=self.device, weights_only=True
        )
        if not isinstance(payload, Mapping):
            raise ValueError("PepDFM parent checkpoint must be a state-dict mapping")
        current = self.model.state_dict()
        if set(payload) != set(current):
            missing = sorted(set(current).difference(payload))
            unexpected = sorted(set(payload).difference(current))
            raise ValueError(
                f"PepDFM parent checkpoint keys differ from the architecture: missing={missing}, unexpected={unexpected}"
            )
        for name, expected in current.items():
            value = payload[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"PepDFM checkpoint value {name!r} is not a tensor")
            if value.shape != expected.shape:
                raise ValueError(
                    f"PepDFM checkpoint tensor {name!r} has shape {tuple(value.shape)}, expected {tuple(expected.shape)}"
                )
            if value.dtype != expected.dtype:
                raise ValueError(
                    f"PepDFM checkpoint tensor {name!r} has dtype {value.dtype}, expected {expected.dtype}"
                )
        self.model.load_state_dict(payload, strict=True)
        self.model.eval()
        return LoadedPepDfmCheckpoint(
            path=str(resolved),
            file_sha256=observed_hash,
            state_dict_sha256=sha256_state_dict(self.model),
        )

    def _tokens(self, sequences: Sequence[str]) -> torch.Tensor:
        encoded = [encode_peptide(sequence) for sequence in sequences]
        if not encoded:
            raise ValueError("at least one sequence is required")
        lengths = {len(row) for row in encoded}
        if len(lengths) != 1:
            raise ValueError("PepDFM replay batches require one fixed sequence length")
        return torch.tensor(encoded, dtype=torch.long, device=self.device)

    def _token_groups(self, sequences: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """Encode a variable-length replay set as homogeneous length groups.

        PepDFM itself supports variable sequence lengths, while an individual
        tensor batch cannot.  Keeping each optimization mini-batch homogeneous
        avoids padding with tokens that the released training objective never
        used.
        """
        grouped: dict[int, list[str]] = {}
        for sequence in sequences:
            grouped.setdefault(len(encode_peptide(sequence)), []).append(sequence)
        if not grouped:
            raise ValueError("at least one sequence is required")
        return tuple(
            (self._tokens(tuple(sorted(grouped[length]))) for length in sorted(grouped))
        )

    def _sample_group_batch(
        self, groups: Sequence[torch.Tensor], *, batch_size: int
    ) -> torch.Tensor:
        """Sample one homogeneous batch, weighting lengths by replay count."""
        weights = torch.tensor(
            [len(group) for group in groups], dtype=torch.float32, device=self.device
        )
        group_index = int(torch.multinomial(weights, 1).item())
        group = groups[group_index]
        indices = torch.randint(
            len(group), (min(batch_size, len(group)),), device=self.device
        )
        return group[indices]

    def _loss(
        self, x_1: torch.Tensor, *, epsilon: float, reduction: str = "mean"
    ) -> torch.Tensor:
        x_0 = torch.randint(
            low=0,
            high=PEPDFM_VOCABULARY_SIZE,
            size=x_1.shape,
            dtype=x_1.dtype,
            device=x_1.device,
        )
        time = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - epsilon)
        path_sample = self.path.sample(t=time, x_0=x_0, x_1=x_1)
        logits = self.model(path_sample.x_t, path_sample.t)
        loss_fn = self._loss_class(path=self.path, reduction=reduction)
        return loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

    def evaluate_losses(
        self,
        sequences: Sequence[str],
        *,
        replicates: int,
        seed: int,
        epsilon: float = 0.001,
    ) -> dict[str, Any]:
        if (
            isinstance(replicates, bool)
            or not isinstance(replicates, int)
            or replicates <= 0
        ):
            raise ValueError("replicates must be a positive integer")
        groups = self._token_groups(sequences)
        random_devices = [self.device] if self.device.type == "cuda" else []
        per_sequence_values: dict[str, float] = {}
        self.model.eval()
        with torch.random.fork_rng(devices=random_devices), torch.no_grad():
            torch.manual_seed(seed)
            for group in groups:
                values: list[torch.Tensor] = []
                for _ in range(replicates):
                    loss = self._loss(group, epsilon=epsilon, reduction="none")
                    while loss.ndim > 1:
                        loss = loss.mean(dim=-1)
                    values.append(loss.detach().cpu())
                group_values = torch.stack(values).mean(dim=0).tolist()
                decoded = [
                    "".join((PEPDFM_TOKEN_BY_ID[int(token)] for token in row[1:-1]))
                    for row in group.detach().cpu().tolist()
                ]
                per_sequence_values.update(zip(decoded, map(float, group_values)))
        ordered = [per_sequence_values[sequence] for sequence in sequences]
        tensor_values = torch.tensor(ordered, dtype=torch.float64)
        return {
            "replicates": replicates,
            "sequence_count": len(sequences),
            "length_group_count": len(groups),
            "mean": float(tensor_values.mean()),
            "standard_deviation_across_sequences": float(
                tensor_values.std(unbiased=False)
            ),
            "per_sequence": dict(zip(sequences, ordered)),
        }

    def sample(
        self,
        *,
        num_samples: int,
        length: int,
        num_steps: int,
        seed: int,
        batch_size: int = 100,
        source_distribution: str = "uniform",
    ) -> tuple[PepDfmSample, ...]:
        if num_samples <= 0 or length <= 0 or num_steps < 2 or (batch_size <= 0):
            raise ValueError("invalid PepDFM sampling dimensions")
        if source_distribution not in {"uniform", "mask"}:
            raise ValueError("source_distribution must be uniform or mask")

        class ProbabilityModelWrapper(self._model_wrapper_class):
            def forward(wrapper_self, x: torch.Tensor, t: torch.Tensor, **extras):
                del extras
                return torch.softmax(wrapper_self.model(x, t), dim=-1)

        solver = self._solver_class(
            model=ProbabilityModelWrapper(self.model),
            path=self.path,
            vocabulary_size=PEPDFM_VOCABULARY_SIZE,
        )
        samples: list[PepDfmSample] = []
        random_devices = [self.device] if self.device.type == "cuda" else []
        self.model.eval()
        with torch.random.fork_rng(devices=random_devices), torch.inference_mode():
            torch.manual_seed(seed)
            while len(samples) < num_samples:
                count = min(batch_size, num_samples - len(samples))
                if source_distribution == "uniform":
                    initial = torch.randint(
                        low=4,
                        high=PEPDFM_VOCABULARY_SIZE,
                        size=(count, length),
                        device=self.device,
                    )
                else:
                    initial = torch.full(
                        (count, length), 3, dtype=torch.long, device=self.device
                    )
                beginning = torch.zeros(
                    (count, 1), dtype=torch.long, device=self.device
                )
                ending = torch.full((count, 1), 2, dtype=torch.long, device=self.device)
                initial = torch.cat([beginning, initial, ending], dim=1)
                solution = solver.sample(
                    x_init=initial,
                    step_size=1.0 / num_steps,
                    verbose=False,
                    time_grid=torch.tensor([0.0, 1.0 - 0.001], device=self.device),
                )
                for row in solution.detach().cpu().tolist():
                    samples.append(
                        _decode_sample(
                            len(samples), tuple((int(token_id) for token_id in row))
                        )
                    )
        return tuple(samples)

    def save_checkpoint(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
