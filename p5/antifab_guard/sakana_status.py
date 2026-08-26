"""
Detector for disabled SAKANA enforcement layers.

The five SAKANA layers protect against data fabrication:
1. Code: numpy RNG hook + template bypass detection
2. Process: critical environment variables force-set at runtime
3. Data: database manager rejects synthetically-marked records
4. Prompt: per-domain "NO SYNTHETIC DATA" hardcoded into agent stages
5. Audit: incident logger records all violations

If any layer is disabled (env vars missing, sentinel files toggled off),
the anti-fabrication guard reports this as a build-failing violation.
"""

import os
from dataclasses import dataclass


@dataclass
class SakanaLayerStatus:
    """Status of all five SAKANA enforcement layers."""

    layer_1_rng_hook: bool
    """Layer 1 (Code): RNG hook active (SYNTHETIC_DATA_FORBIDDEN=true)"""

    layer_2_env_lock: bool
    """Layer 2 (Process): Environment variables enforced"""

    layer_3_database_validation: bool
    """Layer 3 (Data): Database authenticity validator enabled"""

    layer_4_prompt_hardening: bool
    """Layer 4 (Prompt): Agent stage goals include NO_SYNTHETIC constraint"""

    layer_5_incident_logger: bool
    """Layer 5 (Audit): Incident logger active"""

    def any_layer_disabled(self) -> bool:
        """True if any layer is currently disabled."""
        return not all([
            self.layer_1_rng_hook,
            self.layer_2_env_lock,
            self.layer_3_database_validation,
            self.layer_4_prompt_hardening,
            self.layer_5_incident_logger,
        ])

    def disabled_layers(self) -> list[int]:
        """Return list of layer numbers (1–5) that are disabled."""
        disabled = []
        layers = [
            self.layer_1_rng_hook,
            self.layer_2_env_lock,
            self.layer_3_database_validation,
            self.layer_4_prompt_hardening,
            self.layer_5_incident_logger,
        ]
        for i, is_enabled in enumerate(layers, start=1):
            if not is_enabled:
                disabled.append(i)
        return disabled


def check_sakana_status() -> SakanaLayerStatus:
    """
    Inspect the runtime state of all five SAKANA enforcement layers.

    Layer 1 (Code): Checks SYNTHETIC_DATA_FORBIDDEN env var.
    Layer 2 (Process): Checks USE_GLENS_DATA and USE_GEOMIP_DATA (proxy for env enforcement).
    Layer 3 (Data): Would check if database manager is instantiated; simplified here.
    Layer 4 (Prompt): Hardcoded into agent stages; cannot easily inspect at runtime.
    Layer 5 (Audit): Would check incident logger availability; simplified here.

    For layers that are difficult to inspect dynamically, we check the most
    reliable proxy: the environment variables that are set by the enforcement layer.

    Returns:
        SakanaLayerStatus with a boolean for each layer's current state.
    """

    # Layer 1: RNG hook enforces SYNTHETIC_DATA_FORBIDDEN
    layer_1 = os.environ.get("SYNTHETIC_DATA_FORBIDDEN", "").lower() == "true"

    # Layer 2: Process enforcement via env vars
    # (simplified: check if critical vars are set; real enforcement happens
    # inside SECURITY_ENFORCEMENT.py)
    use_glens = os.environ.get("USE_GLENS_DATA", "").lower() == "true"
    use_geomip = os.environ.get("USE_GEOMIP_DATA", "").lower() == "true"
    layer_2 = use_glens and use_geomip

    # Layer 3: Database validation (proxy: assume enabled unless explicitly disabled)
    # In a real system, this would check if the domain_agnostic_database_manager
    # is instantiated and active.
    layer_3_disabled_flag = os.environ.get("DISABLE_DATABASE_VALIDATION", "").lower() == "true"
    layer_3 = not layer_3_disabled_flag

    # Layer 4: Prompt hardening (hardcoded into agent stages)
    # Proxy: check a marker that would be set if agent stages were initialized with constraint
    layer_4_disabled_flag = os.environ.get("DISABLE_PROMPT_HARDENING", "").lower() == "true"
    layer_4 = not layer_4_disabled_flag

    # Layer 5: Incident logger (proxy: assume enabled unless disabled)
    layer_5_disabled_flag = os.environ.get("DISABLE_INCIDENT_LOGGER", "").lower() == "true"
    layer_5 = not layer_5_disabled_flag

    return SakanaLayerStatus(
        layer_1_rng_hook=layer_1,
        layer_2_env_lock=layer_2,
        layer_3_database_validation=layer_3,
        layer_4_prompt_hardening=layer_4,
        layer_5_incident_logger=layer_5,
    )
