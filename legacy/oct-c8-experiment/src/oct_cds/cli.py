"""Small CLI wrapper. Training itself is Hydra-driven via ``train.py``.

    python -m oct_cds.cli data build                 # build all manifests
    python -m oct_cds.cli data build --only oct_c8
    python -m oct_cds.cli cds demo                   # run the rule engine on a fake case
"""

from __future__ import annotations

import argparse
import sys

from omegaconf import OmegaConf

from oct_cds.common.logging import get_logger
from oct_cds.common.paths import REPO_ROOT

log = get_logger("cli")
CONFIG_DIR = REPO_ROOT / "configs" / "data"
PATHS_DIR = REPO_ROOT / "configs" / "paths"

# ${hydra:runtime.cwd} only exists under a Hydra run; outside it (this CLI) the
# repo root is the right anchor. Registered once, idempotently.
if not OmegaConf.has_resolver("hydra"):
    OmegaConf.register_new_resolver(
        "hydra", lambda key: str(REPO_ROOT) if key == "runtime.cwd" else key
    )


def _load_data_cfg(
    name: str, overrides: list[str] | None = None, paths_name: str = "default"
) -> dict:
    """Compose  paths/<paths_name>.yaml + data/<name>.yaml  exactly the way Hydra
    would (data config sees ``${paths.*}``), then fully resolve interpolations.

    ``paths_name`` selects the environment (e.g. "kaggle") — same as Hydra's
    ``paths=<name>`` group override. ``overrides`` are dotlist strings like
    ``paths.oct_c8_raw_root=/some/path``.
    """
    paths_file = PATHS_DIR / f"{paths_name}.yaml"
    if not paths_file.exists():
        raise SystemExit(
            f"unknown paths config {paths_name!r}: {paths_file} not found "
            f"(available: {sorted(p.stem for p in PATHS_DIR.glob('*.yaml'))})"
        )
    cfg = OmegaConf.create(
        {
            "paths": OmegaConf.load(paths_file),
            "data": OmegaConf.load(CONFIG_DIR / f"{name}.yaml"),
        }
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    resolved = OmegaConf.to_container(cfg, resolve=True)  # raises on bad interpolation
    return resolved["data"]


def cmd_data_build(args: argparse.Namespace) -> int:
    from oct_cds.data.manifest import (
        assert_no_leakage,
        build_oct_c8_manifest,
        build_optopol_manifest,
        write_manifests,
    )

    targets = args.only or ["oct_c8", "clinic_optopol"]
    overrides = args.set or []
    paths_name = args.paths
    manifests: dict = {}

    if "oct_c8" in targets:
        cfg = _load_data_cfg("oct_c8", overrides, paths_name)
        log.info("oct_c8 root -> %s", cfg["root"])
        m = build_oct_c8_manifest(cfg, probe_images=not args.fast)
        write_manifests(m, cfg)
        manifests.update(m)

    if "clinic_optopol" in targets:
        cfg = _load_data_cfg("clinic_optopol", overrides, paths_name)
        log.info("clinic_optopol root -> %s", cfg["root"])
        df = build_optopol_manifest(cfg, probe_images=not args.fast)
        write_manifests({"external_test": df}, cfg)
        manifests["external_test"] = df

    assert_no_leakage(manifests)
    log.info("manifests OK: %s", {k: len(v) for k, v in manifests.items()})
    return 0


def cmd_cds_demo(_: argparse.Namespace) -> int:
    from oct_cds.cds.report import build_report
    from oct_cds.cds.rules import CDSRuleEngine
    from oct_cds.cds.schema import CaseInput, ModelResult

    mr = ModelResult(
        probs={
            "AMD": 0.03, "CNV": 0.78, "CSR": 0.02, "DME": 0.05,
            "DR": 0.02, "Drusen": 0.03, "Macular Hole": 0.02, "Normal": 0.05,
        },
        ood_score=0.22,
        model_version="densenet121@demo",
    )
    case = CaseInput(image_path="demo.png", eye="OD", symptoms=["acute distortion"])
    rec = CDSRuleEngine().evaluate(mr, case)
    print(build_report(rec, mr, case)["narrative"])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oct-cds")
    sub = p.add_subparsers(dest="group", required=True)

    d = sub.add_parser("data").add_subparsers(dest="action", required=True)
    db = d.add_parser("build")
    db.add_argument("--only", nargs="*", choices=["oct_c8", "clinic_optopol"])
    db.add_argument("--fast", action="store_true", help="skip md5 / size probing")
    db.add_argument(
        "--paths",
        default="default",
        metavar="NAME",
        help="paths config to use (configs/paths/<NAME>.yaml), e.g. --paths kaggle",
    )
    db.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        help="config overrides, e.g. --set paths.oct_c8_raw_root=/content/drive/.../RetinalOCT_Dataset",
    )
    db.set_defaults(func=cmd_data_build)

    c = sub.add_parser("cds").add_subparsers(dest="action", required=True)
    c.add_parser("demo").set_defaults(func=cmd_cds_demo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
