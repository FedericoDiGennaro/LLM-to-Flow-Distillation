"""Portable fixed-pool evaluation with released MOG-DFM peptide predictors.

The upstream inference scripts contain author-specific absolute paths and load
one ESM-2 backbone per objective. This adapter resolves the pinned checkpoints
explicitly and shares the ESM-2 weights embedded in the released affinity
checkpoint across the four developability heads. It is an evaluation adapter;
integration into every MOG-DFM sampling step is intentionally a later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from domains.peptides.model import PEPDFM_TOKEN_BY_ID, sha256_file


ESM2_REPO_ID = "facebook/esm2_t33_650M_UR50D"
ESM2_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
# Architecture metadata from the pinned Hugging Face ESM-2 config.  The
# released MOG-DFM affinity checkpoint already embeds the complete ESM-2
# weights, so fetching model metadata at inference time is both unnecessary
# and incompatible with our deliberately offline Slurm jobs.
ESM2_CONFIG = {
    "architectures": ["EsmForMaskedLM"],
    "attention_probs_dropout_prob": 0.0,
    "classifier_dropout": None,
    "emb_layer_norm_before": False,
    "esmfold_config": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 1280,
    "initializer_range": 0.02,
    "intermediate_size": 5120,
    "is_folding_model": False,
    "layer_norm_eps": 1e-5,
    "mask_token_id": 32,
    "max_position_embeddings": 1026,
    "model_type": "esm",
    "num_attention_heads": 20,
    "num_hidden_layers": 33,
    "pad_token_id": 1,
    "position_embedding_type": "rotary",
    "token_dropout": True,
    "use_cache": True,
    "vocab_list": None,
    "vocab_size": 33,
}
MOG_DFM_PREDICTOR_PATHS = {
    "affinity": Path("classifier_ckpt/binding_affinity_unpooled.pt"),
    "hemolysis": Path("classifier_ckpt/best_model_hemolysis.json"),
    "non_fouling": Path("classifier_ckpt/wt_nonfouling.pt"),
    "solubility": Path("classifier_ckpt/best_model_solubility.json"),
    "half_life": Path("classifier_ckpt/wt_halflife.pt"),
    "affinity_code": Path("classifier_code/binding_affinity_unpooled.py"),
    "non_fouling_code": Path("classifier_code/nonfouling_wt.py"),
    "half_life_code": Path("classifier_code/half_life.py"),
}
MOG_DFM_PREDICTOR_SHA256 = {
    "affinity": "fc28ae9f09b981b07547a773ca2e07f241cb08b3b8aa901e66627ff153f3aa8b",
    "hemolysis": "09b90730272d48f061bf41c79a5ae44f5c977f331d48600c8615852806308be1",
    "non_fouling": "d0fce9a0e94da09adfa19e430c2f0ca2de34ae3ce3049c86b51268c908ee44bc",
    "solubility": "948ac245c158aacd51f36cc71f4ee7bbbf3568c92666c637689790a01677fa59",
    "half_life": "d05a90b794ed45246cf2778e677f5ef312db90131c0b39367b3102fde59b09fa",
}


@dataclass(frozen=True)
class MogDfmPredictorAssets:
    """Resolved released checkpoints and inference definitions."""

    root: Path
    paths: dict[str, Path]

    @classmethod
    def resolve(
        cls, root: Path, *, verify_hashes: bool = True
    ) -> "MogDfmPredictorAssets":
        root = root.resolve()
        paths = {
            name: root / relative
            for name, relative in MOG_DFM_PREDICTOR_PATHS.items()
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "MOG-DFM predictor assets are incomplete; run "
                "domains.peptides.fetch_assets. Missing: " + ", ".join(missing)
            )
        if verify_hashes:
            for name, expected in MOG_DFM_PREDICTOR_SHA256.items():
                observed = sha256_file(paths[name])
                if observed != expected:
                    raise ValueError(
                        f"MOG-DFM {name} checkpoint hash mismatch: "
                        f"expected {expected}, observed {observed}"
                    )
        return cls(root=root, paths=paths)


def encode_esm2_sequence(
    sequence: str, *, device: torch.device | None = None
) -> Tensor:
    """Encode canonical amino-acid text using PepDFM's ESM-2 token IDs."""
    sequence = sequence.strip().upper()
    token_id = {token: index for index, token in enumerate(PEPDFM_TOKEN_BY_ID)}
    invalid = sorted(set(sequence).difference(token_id))
    if not sequence or invalid:
        detail = ", ".join(invalid) if invalid else "empty sequence"
        raise ValueError(f"sequence is not canonical amino-acid text: {detail}")
    values = [
        token_id["<cls>"],
        *(token_id[residue] for residue in sequence),
        token_id["<eos>"],
    ]
    return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0)


def _load_upstream_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load upstream module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_offline_esm2_factories(
    affinity_module: ModuleType,
    *,
    config: Any,
    model_type: type[nn.Module],
) -> None:
    """Redirect both upstream Hugging Face factories to pinned local objects."""

    class LocalAutoModel:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> nn.Module:
            del args, kwargs
            return model_type(config)

    class LocalAutoConfig:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return config

    # The released class calls both names from its own imported module.  Both
    # must be rebound before construction or an offline run still attempts a
    # Hugging Face metadata lookup after creating the local model.
    affinity_module.AutoModel = LocalAutoModel
    affinity_module.AutoConfig = LocalAutoConfig


def _clean_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


@dataclass(frozen=True)
class MogDfmScoreBatch:
    """Five high-is-better released objective values for one peptide batch."""

    affinity: Tensor
    non_hemolysis: Tensor
    non_fouling: Tensor
    solubility: Tensor
    half_life_hours: Tensor

    def as_columns(self) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in tensor.detach().cpu()]
            for name, tensor in (
                ("affinity", self.affinity),
                ("non_hemolysis", self.non_hemolysis),
                ("non_fouling", self.non_fouling),
                ("solubility", self.solubility),
                ("half_life_hours", self.half_life_hours),
            )
        }


@dataclass(frozen=True)
class MogDfmDevelopabilityBatch:
    """Four target-independent released predictor outputs for one batch."""

    non_hemolysis: Tensor
    non_fouling: Tensor
    solubility: Tensor
    half_life_hours: Tensor

    def as_columns(self) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in tensor.detach().cpu()]
            for name, tensor in (
                ("non_hemolysis", self.non_hemolysis),
                ("non_fouling", self.non_fouling),
                ("solubility", self.solubility),
                ("half_life_hours", self.half_life_hours),
            )
        }


class MogDfmPredictorSuite(nn.Module):
    """Load and evaluate the five released MOG-DFM peptide objectives."""

    objective_names = (
        "affinity",
        "non_hemolysis",
        "non_fouling",
        "solubility",
        "half_life_hours",
    )

    def __init__(
        self,
        assets: MogDfmPredictorAssets,
        target_sequence: str,
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        try:
            import xgboost as xgb
            from transformers import EsmConfig, EsmModel
        except ImportError as error:
            raise RuntimeError(
                "released MOG-DFM predictors require xgboost and transformers"
            ) from error

        affinity_module = _load_upstream_module(
            "mogdfm_upstream_affinity", assets.paths["affinity_code"]
        )
        config = EsmConfig(**ESM2_CONFIG)

        _bind_offline_esm2_factories(
            affinity_module,
            config=config,
            model_type=EsmModel,
        )
        affinity = affinity_module.UnpooledBindingPredictor(
            esm_model_name=ESM2_REPO_ID,
            hidden_dim=384,
            kernel_sizes=[3, 5, 7],
            n_heads=8,
            n_layers=4,
            dropout=0.14561457009902096,
            freeze_esm=True,
        )
        affinity_checkpoint = torch.load(
            assets.paths["affinity"], map_location="cpu", weights_only=False
        )
        compatibility = affinity.load_state_dict(
            affinity_checkpoint["model_state_dict"], strict=False
        )
        allowed_unexpected = {
            "esm_model.embeddings.position_embeddings.weight"
        }
        if compatibility.missing_keys or set(compatibility.unexpected_keys) != allowed_unexpected:
            raise RuntimeError(
                "released affinity checkpoint is incompatible with the current "
                f"Transformers architecture: {compatibility}"
            )
        self.affinity = affinity.to(device).eval()
        for parameter in self.affinity.parameters():
            parameter.requires_grad = False

        # The affinity checkpoint embeds the complete ESM-2 backbone. Reusing it
        # avoids four redundant multi-gigabyte backbone allocations.
        self.encoder = self.affinity.esm_model
        self.hemolysis = xgb.Booster(model_file=str(assets.paths["hemolysis"]))
        self.solubility = xgb.Booster(model_file=str(assets.paths["solubility"]))

        non_fouling_module = _load_upstream_module(
            "mogdfm_upstream_non_fouling", assets.paths["non_fouling_code"]
        )
        non_fouling_checkpoint = torch.load(
            assets.paths["non_fouling"], map_location="cpu", weights_only=False
        )
        params = non_fouling_checkpoint["best_params"]
        self.non_fouling_head = non_fouling_module.MLPClassifier(
            in_dim=1280,
            hidden=int(params["hidden"]),
            dropout=float(params.get("dropout", 0.1)),
        ).to(device)
        self.non_fouling_head.load_state_dict(
            non_fouling_checkpoint["state_dict"]
        )
        self.non_fouling_head.eval()

        half_life_module = _load_upstream_module(
            "mogdfm_upstream_half_life", assets.paths["half_life_code"]
        )
        half_life_checkpoint = torch.load(
            assets.paths["half_life"], map_location="cpu", weights_only=False
        )
        self.half_life_head = half_life_module.build_model(
            "transformer",
            int(half_life_checkpoint["in_dim"]),
            half_life_checkpoint.get("best_params", {}),
        ).to(device)
        self.half_life_head.load_state_dict(
            _clean_state_dict(half_life_checkpoint["state_dict"]), strict=True
        )
        self.half_life_head.eval()
        self.half_life_target_col = half_life_checkpoint.get(
            "target_col", "label"
        )
        self.register_buffer(
            "target_tokens", encode_esm2_sequence(target_sequence, device=device)
        )

    @property
    def device(self) -> torch.device:
        return self.target_tokens.device

    @torch.inference_mode()
    def developability(self, tokens: Tensor) -> MogDfmDevelopabilityBatch:
        """Evaluate only target-independent heads; do not compute affinity."""
        if tokens.ndim != 2 or tokens.shape[1] < 3:
            raise ValueError("tokens must have shape [batch, length >= 3]")
        tokens = tokens.to(self.device)
        attention = torch.ones_like(tokens)
        hidden = self.encoder(
            input_ids=tokens, attention_mask=attention
        ).last_hidden_state

        pooled = hidden.mean(dim=1).float().cpu().numpy()
        pooled = np.nan_to_num(pooled, nan=0.0)
        pooled = np.clip(
            pooled, np.finfo(np.float32).min, np.finfo(np.float32).max
        )
        import xgboost as xgb

        matrix = xgb.DMatrix(pooled)
        non_hemolysis = 1.0 - self.hemolysis.predict(matrix)
        solubility = self.solubility.predict(matrix)

        residue_mask = (tokens != 0) & (tokens != 1) & (tokens != 2)
        non_fouling = torch.sigmoid(
            self.non_fouling_head(hidden, residue_mask)
        )

        core_hidden = hidden[:, 1:-1, :].float()
        core_mask = torch.ones(
            core_hidden.shape[:2], dtype=torch.bool, device=self.device
        )
        half_life = self.half_life_head(core_hidden, core_mask)
        if self.half_life_target_col == "log_label":
            half_life = torch.expm1(half_life)

        return MogDfmDevelopabilityBatch(
            non_hemolysis=torch.from_numpy(non_hemolysis).to(self.device),
            non_fouling=non_fouling,
            solubility=torch.from_numpy(solubility).to(self.device),
            half_life_hours=half_life,
        )

    @torch.inference_mode()
    def forward(self, tokens: Tensor) -> MogDfmScoreBatch:
        developability = self.developability(tokens)
        tokens = tokens.to(self.device)
        target = self.target_tokens.repeat(tokens.shape[0], 1)
        affinity_output = self.affinity(
            protein_input_ids=target,
            binder_input_ids=tokens,
        )
        affinity = affinity_output[0].squeeze(-1)

        return MogDfmScoreBatch(
            affinity=affinity,
            non_hemolysis=developability.non_hemolysis,
            non_fouling=developability.non_fouling,
            solubility=developability.solubility,
            half_life_hours=developability.half_life_hours,
        )
