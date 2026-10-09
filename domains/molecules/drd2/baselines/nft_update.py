"""DiffusionNFT on FlowMol3's flow-matching vector field (arXiv:2509.16117).

The paper is written for a velocity predictor. FlowMol3 *is* a velocity /
endpoint predictor, so the implicit-policy construction applies directly to
``vector_field`` outputs. Coordinate, atom-type, charge and bond heads are
combined with FlowMol's own loss weighting, matching the flow-DPO adapter.

``reference`` is the data-collection policy that produced the rollout, not a
fixed pretrained anchor: it is EMA-updated between iterations. One invocation
of the campaign trainer is one Algorithm-1 iteration.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from domains.molecules.drd2.dpo import copy_dgl_graph, per_graph_flow_loss
from domains.molecules.drd2.baselines.nft_loss import implicit_policies, negative_aware_loss


def implicit_feature_policies(
    policy_output: Mapping[str, Tensor],
    reference_output: Mapping[str, Tensor],
    *,
    beta: float,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Form implicit positive/negative predictions for every FlowMol head."""

    if set(policy_output) != set(reference_output):
        raise ValueError("policy and reference FlowMol heads must match")
    if not policy_output:
        raise ValueError("FlowMol NFT needs at least one vector-field head")
    positive: dict[str, Tensor] = {}
    negative: dict[str, Tensor] = {}
    for name, policy_tensor in policy_output.items():
        pos, neg = implicit_policies(
            policy_tensor, reference_output[name], beta=beta
        )
        positive[name] = pos
        negative[name] = neg
    return positive, negative


def flowmol_nft_step(
    policy: Any,
    reference: Any,
    interpolated: Any,
    times: Tensor,
    *,
    optimality: Tensor,
    beta: float,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Evaluate one DiffusionNFT step on a shared FlowMol interpolant."""

    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    optimality = optimality.reshape(-1)
    batch_size = int(interpolated.batch_size)
    if optimality.numel() != batch_size:
        raise ValueError("DiffusionNFT needs one optimality probability per molecule")
    # Clone both before either forward pass: vector fields may mutate graphs.
    policy_graph = copy_dgl_graph(interpolated)
    reference_graph = copy_dgl_graph(interpolated)
    node_batch_idx, _edge_batch_idx = get_batch_idxs(policy_graph)
    upper_edge_mask = get_upper_edge_mask(policy_graph)
    policy_output = policy.vector_field(
        policy_graph,
        times,
        node_batch_idx=node_batch_idx,
        upper_edge_mask=upper_edge_mask,
    )
    with torch.no_grad():
        reference_nodes, _ = get_batch_idxs(reference_graph)
        reference_edges = get_upper_edge_mask(reference_graph)
        reference_output = reference.vector_field(
            reference_graph,
            times,
            node_batch_idx=reference_nodes,
            upper_edge_mask=reference_edges,
        )
    positive_output, negative_output = implicit_feature_policies(
        policy_output, reference_output, beta=beta
    )
    positive_loss = per_graph_flow_loss(
        policy, policy_graph, times, vf_output=positive_output
    )
    negative_loss = per_graph_flow_loss(
        policy, policy_graph, times, vf_output=negative_output
    )
    loss = negative_aware_loss(
        positive_loss, negative_loss, optimality.to(positive_loss)
    )
    with torch.no_grad():
        reference_loss = per_graph_flow_loss(
            reference, reference_graph, times, vf_output=reference_output
        )
    return loss, {
        "positive_branch_loss": positive_loss.detach(),
        "negative_branch_loss": negative_loss.detach(),
        "reference_flow_loss": reference_loss.detach(),
        "optimality": optimality.detach(),
    }


__all__ = [
    "flowmol_nft_step",
    "implicit_feature_policies",
]
