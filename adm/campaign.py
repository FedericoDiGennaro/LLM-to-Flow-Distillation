"""Fail-closed access to single- or multi-experiment campaign registrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CampaignRegistration:
    """The resource and approval records authorized for one experiment."""

    experiment_id: str
    resources: Mapping[str, Any]
    approval_gate: Mapping[str, Any]


def resolve_campaign_registration(
    campaign: Mapping[str, Any], *, experiment_id: str
) -> CampaignRegistration:
    """Resolve one experiment without permitting cross-arm resource fallback.

    Legacy campaigns bind a single top-level ``experiment_id``, ``resources``,
    and ``approval_gate``.  Umbrella campaigns instead bind an
    ``experiment_registrations`` object.  The two layouts are deliberately
    mutually exclusive so an incomplete migration cannot silently select the
    wrong output root, config, authorization, or compute resources.
    """

    if not experiment_id:
        raise PermissionError("campaign registration requires an experiment_id")
    registrations = campaign.get("experiment_registrations")
    if registrations is None:
        if campaign.get("experiment_id") != experiment_id:
            raise PermissionError("campaign does not authorize this experiment")
        resources = campaign.get("resources")
        approval_gate = campaign.get("approval_gate")
    else:
        if any(
            key in campaign for key in ("experiment_id", "resources", "approval_gate")
        ):
            raise PermissionError(
                "umbrella campaign cannot mix top-level and per-experiment registrations"
            )
        if not isinstance(registrations, Mapping):
            raise PermissionError("campaign experiment registrations are malformed")
        registration = registrations.get(experiment_id)
        if not isinstance(registration, Mapping):
            raise PermissionError("campaign has no registration for this experiment")
        if registration.get("experiment_id") != experiment_id:
            raise PermissionError("campaign registration identity differs")
        resources = registration.get("resources")
        approval_gate = registration.get("approval_gate")
    if not isinstance(resources, Mapping):
        raise PermissionError("campaign resource binding is missing")
    if not isinstance(approval_gate, Mapping):
        raise PermissionError("campaign approval gate is missing")
    return CampaignRegistration(
        experiment_id=experiment_id,
        resources=resources,
        approval_gate=approval_gate,
    )


__all__ = ["CampaignRegistration", "resolve_campaign_registration"]
