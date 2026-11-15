"""Generate Nexys A7 (Artix-7) sequential UART RTL and a Vivado TCL project.

Reuses the *existing local* Lattice generator's layout parsing and explicit
SystemVerilog placeholder rendering. Does not modify Lattice source/output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys
from typing import Any


def _load_render_helpers(repo_root: Path):
    path = repo_root / "src" / "tcl" / "remodel_lattice_project.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared rendering helpers: {path}")
    name = "_nexys_shared_lattice_render"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_project(
    repo_root: str | Path,
    mem_layout_path: str | Path,
    *,
    fpga_mode: str = "sequential",
    template_rel: str = "src/rtl/src_nexys_sequential",
    generated_rel: str = "src/rtl/generated/src_nexys_sequential",
    project_name: str = "SOC_MLP_nexys_a7_sequential",
    part_name: str = "xc7a100tcsg324-1",
    jobs: int = 8,
    dsp_pw: int = 48,
    clean_generated: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if fpga_mode != "sequential":
        raise NotImplementedError("Nexys generator currently supports sequential mode only")
    shared = _load_render_helpers(root)
    info = shared.parse_layout(mem_layout_path, mode_override=fpga_mode)
    template_dir = root / template_rel
    out_dir = root / generated_rel
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Nexys RTL template directory missing: {template_dir}")

    required = list(shared.LATTICE_RTL_FILES_SEQUENTIAL)
    missing = [name for name in required if not (template_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RTL templates: {missing}")

    if clean_generated and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales_path = shared.sync_scales_vh(template_dir, out_dir, info.mem_dir)
    scales = shared.validate_scales_against_layout(scales_path, info)
    # Numerical contract: the generated header and the profile must agree.
    header_width = int(scales["DSP_PW"])
    if header_width != int(dsp_pw):
        raise ValueError(
            f"DSP width mismatch: scales.vh has {header_width}, "
            f"but Nexys target requires {dsp_pw}. Re-export the model for Nexys."
        )
    substitutions = shared.build_sequential_sv_variables(info, scales, dsp_pw=dsp_pw)
    common = substitutions.get("common", {})
    for name in required:
        source = (template_dir / name).read_text(encoding="utf-8")
        fields = dict(common)
        fields.update(substitutions.get(name, {}))
        rendered = shared.render_sv_placeholders(source, fields, filename=name)
        (out_dir / name).write_text(rendered, encoding="utf-8")

    # Render any extra SystemVerilog helper included in the Nexys template.
    for source_path in sorted(template_dir.glob("*.sv")):
        if source_path.name in required:
            continue
        fields = dict(common)
        fields.update(substitutions.get(source_path.name, {}))
        rendered = shared.render_sv_placeholders(
            source_path.read_text(encoding="utf-8"), fields, filename=source_path.name
        )
        (out_dir / source_path.name).write_text(rendered, encoding="utf-8")

    xdc = root / "constraints" / "nexys_a7.xdc"
    if not xdc.is_file():
        raise FileNotFoundError(
            f"Missing Nexys A7 constraints: {xdc}. Create a real XDC for clk, uart_rx, uart_tx."
        )
