"""Guided-wave attenuation imaging for CFRP-steel interfacial debonding."""

from .config import DEFAULT_CONFIG, PipelineConfig
from .pipeline import ImagingResult, run_imaging

__all__ = ["DEFAULT_CONFIG", "PipelineConfig", "ImagingResult", "run_imaging"]
from .sensors import manuscript_sensor_layout

__all__.append("manuscript_sensor_layout")
