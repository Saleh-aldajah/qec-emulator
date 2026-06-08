"""
config.py — YAML/JSON configuration management.

Supports file-based and programmatic configuration with validation,
defaults, and environment variable overrides. Analogous to Cooja's
cooja.config / simulation XML approach but in a structured, typed form.

Author: Dr. Saleh H. AlDaajeh  ORCID: 0000-0001-7810-9290
"""
from __future__ import annotations
import json, os, copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    _YAML = True
except ImportError:
    _YAML = False


# ---------------------------------------------------------------------------
# SimulationConfig  — one run configuration
# ---------------------------------------------------------------------------

@dataclass
class CodeConfig:
    family:     str   = "BB"          # "BB" | "Steane7" | "custom"
    ell:        int   = 6
    m:          int   = 6
    a_exponents: List[int] = field(default_factory=lambda: [3, 1, 2])
    b_exponents: List[int] = field(default_factory=lambda: [3, 1, 2])
    # convenience presets: "BB72" | "BB144" | "BB288"
    preset:     Optional[str] = None


@dataclass
class NoiseConfig:
    model:      str   = "phenomenological"  # "phenomenological"|"circuit"|"hardware_cz"|"hardware_ms"
    p_values:   List[float] = field(default_factory=lambda: [0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012])
    rounds:     int   = 1         # >1 activates repeated-cycle model
    p_idle:     float = 0.0
    p_meas:     float = 0.0


@dataclass
class DecoderConfig:
    name:       str   = "bposd"   # "bposd"|"mwpm"|"uf"|"lookup"
    bp_method:  str   = "ms"
    ms_scaling_factor: float = 0.625
    osd_order:  int   = 7
    max_iter:   int   = 100


@dataclass
class RunnerConfig:
    mode:       str   = "sweep"   # "sweep"|"threshold"|"repeated_cycle"|"certificate"|"hardware_ablation"
    trials:     int   = 5000
    seed:       int   = 20260607
    batches:    int   = 1
    verbose:    bool  = True
    parallel:   bool  = False
    workers:    int   = 4


@dataclass
class ExportConfig:
    formats:    List[str] = field(default_factory=lambda: ["csv"])  # csv|json|hdf5
    output_dir: str   = "results"
    prefix:     str   = "run"
    overwrite:  bool  = True


@dataclass
class ServerConfig:
    host:       str   = "0.0.0.0"
    port:       int   = 8765
    reload:     bool  = False
    log_level:  str   = "info"


@dataclass
class SimulationConfig:
    """
    Top-level configuration object. Can be loaded from YAML/JSON
    or constructed programmatically.

    Example YAML::

        code:
          preset: BB72
        noise:
          model: phenomenological
          p_values: [0.001, 0.004, 0.008, 0.012]
        decoder:
          name: bposd
        runner:
          trials: 10000
          parallel: true
          workers: 8
        export:
          formats: [csv, json]
          output_dir: my_results
    """
    code:    CodeConfig    = field(default_factory=CodeConfig)
    noise:   NoiseConfig   = field(default_factory=NoiseConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    runner:  RunnerConfig  = field(default_factory=RunnerConfig)
    export:  ExportConfig  = field(default_factory=ExportConfig)
    server:  ServerConfig  = field(default_factory=ServerConfig)

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        if not _YAML:
            raise ImportError("pyyaml is required for YAML export")
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.suffix in ('.yaml', '.yml'):
            path.write_text(self.to_yaml())
        else:
            path.write_text(self.to_json())

    # ── Deserialisation ────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        def _merge(dc_cls, data):
            import dataclasses
            if not isinstance(data, dict):
                return data
            kwargs = {}
            for f in dataclasses.fields(dc_cls):
                if f.name in data:
                    val = data[f.name]
                    # recurse into nested dataclasses
                    ft = f.type
                    nested = {
                        'CodeConfig': CodeConfig, 'NoiseConfig': NoiseConfig,
                        'DecoderConfig': DecoderConfig, 'RunnerConfig': RunnerConfig,
                        'ExportConfig': ExportConfig, 'ServerConfig': ServerConfig,
                    }.get(ft if isinstance(ft, str) else ft.__name__ if hasattr(ft,'__name__') else '', None)
                    if nested and isinstance(val, dict):
                        kwargs[f.name] = _merge(nested, val)
                    else:
                        kwargs[f.name] = val
            return dc_cls(**kwargs)
        return _merge(cls, d)

    @classmethod
    def from_json(cls, text: str) -> "SimulationConfig":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> "SimulationConfig":
        if not _YAML:
            raise ImportError("pyyaml required")
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SimulationConfig":
        """Load from .yaml/.yml or .json file with env-var overrides."""
        path = Path(path)
        text = path.read_text()
        cfg  = cls.from_yaml(text) if path.suffix in ('.yaml','.yml') else cls.from_json(text)
        return cfg._apply_env_overrides()

    def _apply_env_overrides(self) -> "SimulationConfig":
        """
        Allow environment variables to override config values.

        QEC_TRIALS=20000       → runner.trials
        QEC_WORKERS=8          → runner.workers
        QEC_PORT=9000          → server.port
        QEC_OUTPUT_DIR=/tmp    → export.output_dir
        """
        mapping = {
            'QEC_TRIALS':     ('runner', 'trials',     int),
            'QEC_WORKERS':    ('runner', 'workers',    int),
            'QEC_PORT':       ('server', 'port',       int),
            'QEC_OUTPUT_DIR': ('export', 'output_dir', str),
            'QEC_SEED':       ('runner', 'seed',       int),
            'QEC_VERBOSE':    ('runner', 'verbose',    lambda x: x.lower() in ('1','true','yes')),
        }
        for env_var, (section, key, cast) in mapping.items():
            val = os.environ.get(env_var)
            if val is not None:
                setattr(getattr(self, section), key, cast(val))
        return self


# ---------------------------------------------------------------------------
# Default config file
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_YAML = """\
# QEC Emulator default configuration
# See qec_emulator.config.SimulationConfig for full schema

code:
  preset: BB72           # BB72 | BB144 | BB288 | null (use ell/m/exponents)

noise:
  model: phenomenological
  p_values: [0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
  rounds: 1

decoder:
  name: bposd
  bp_method: ms
  ms_scaling_factor: 0.625
  osd_order: 7
  max_iter: 100

runner:
  mode: sweep
  trials: 5000
  seed: 20260607
  parallel: false
  workers: 4
  verbose: true

export:
  formats: [csv]
  output_dir: results
  prefix: run
  overwrite: true

server:
  host: 0.0.0.0
  port: 8765
"""


def write_default_config(path: Union[str, Path] = "qec_config.yaml") -> Path:
    """Write the default configuration to a file and return the path."""
    p = Path(path)
    p.write_text(DEFAULT_CONFIG_YAML)
    return p
