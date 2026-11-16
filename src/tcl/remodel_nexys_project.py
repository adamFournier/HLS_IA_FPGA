"""Standalone Nexys A7 / Vivado generator for sequential and pipeline UART MLP.

This file is independent of remodel_lattice_project.py and any Lattice module.
It converts a quantization export (mem_layout.json, scales.vh, *.mem) into
concrete SystemVerilog and a Vivado Tcl project for an Artix-7 Nexys A7.

Template layout:
    src/rtl/src_nexys_sequential/*.sv
Quantized export:
    data/quantized_data/<name>/nexys_a7/sequential/{mem_layout.json,scales.vh,*.mem}
Outputs:
    src/rtl/generated/src_nexys_sequential/*.sv
    vivado/create_soc_mlp_project_nexys.tcl

Only the sequential architecture is supported. The project must provide an
actual board-specific XDC with clk, uart_rx, uart_tx pin constraints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable


NEXYS_RTL_FILES_SEQUENTIAL = [
    "dsp.sv",
    "mem_a.sv",
    "mem_b.sv",
    "rom_weight_all.sv",
    "rom_bias_all.sv",
    "rom_act_all.sv",
    "mlp_sequential_ctrl.sv",
    "mlp_sequential_datapath.sv",
    "mlp_sequential.sv",
    "uart_rx.sv",
    "uart_tx.sv",
    "uart_input.sv",
    "uart_output.sv",
    "soc_mlp.sv",
]

NEXYS_RTL_FILES_PIPELINE_STATIC = [
    "dsp.sv", "layer_ctrl.sv", "layer_datapath.sv", "phase_ctrl.sv",
    "pp_buffer.sv", "mlp_pipeline.sv", "uart_rx.sv", "uart_tx.sv",
    "uart_input.sv", "uart_output.sv", "soc_mlp.sv",
]
NEXYS_PIPELINE_PER_LAYER_TEMPLATES = (
    "layer.sv", "rom_weight.sv", "rom_bias.sv", "rom_act.sv"
)

KNOWN_PLACEHOLDERS = {
    "memory_weight", "memory_bias", "memory_act_function",
    "layer_size_case", "weight_base_case", "bias_base_case",
    "act_depth_localparams", "act_depth_sum_expr",
    "sx_case", "sw_case", "sb_case", "sp_case", "act_case",
    "lutn_case", "act_base_case", "weight_depth_expr",
    "bias_depth_expr", "dsp_pw_expr", "output_size_expr", "max_neurons_expr",
    "module_layer_name", "module_weight_name", "module_bias_name", "module_act_name",
    "n_tab_expr", "sx_tab_expr", "sw_tab_expr", "sb_tab_expr",
    "sp_tab_expr", "act_tab_expr", "lutn_tab_expr",
    "intermediate_buffers", "layer_instances",
}


@dataclass(frozen=True)
class LayoutInfo:
    layout_path: Path
    mem_dir: Path
    mode: str
    num_linear: int
    topology: list[int]
    weight_file: str
    bias_file: str
    act_file: str
    total_weight_words: int
    total_bias_words: int
    total_activation_words: int
    weights: list[dict[str, Any]] = field(default_factory=list)
    biases: list[dict[str, Any]] = field(default_factory=list)
    activations: list[dict[str, Any]] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_mode(layout: dict[str, Any], mode_override: str | None = None) -> str:
    if mode_override is not None:
        mode = mode_override.strip().lower()
    else:
        raw = str(layout.get("mode") or layout.get("format") or "").lower()
        mode = "pipeline" if "pipeline" in raw else "sequential"
    if mode in {"sequential", "seq", "concat"}:
        return "sequential"
    if mode in {"pipeline", "pipelined"}:
        return "pipeline"
    raise ValueError(f"Unsupported architecture: {mode!r}")


def _file_basename(value: Any, default: str) -> str:
    return Path(str(value)).name if value else default


def parse_layout(layout_path: str | Path, mode_override: str | None = None) -> LayoutInfo:
    path = Path(layout_path).resolve()
    layout = _read_json(path)
    mode = infer_mode(layout, mode_override)

    weights = sorted(layout.get("weights", []), key=lambda w: int(w.get("layer_idx", 0)))
    biases = sorted(layout.get("biases", []), key=lambda b: int(b.get("layer_idx", 0)))
    acts = sorted(layout.get("activations", []), key=lambda a: int(a.get("layer_idx", 0)))
    if not weights:
        raise ValueError("mem_layout.json must contain a nonempty weights section")

    topology: list[int] = []
    for index, weight in enumerate(weights):
        shape = weight.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError(f"weights[{index}] must have shape=[out_features,in_features]")
        n_out, n_in = map(int, shape)
        if min(n_out, n_in) <= 0:
            raise ValueError(f"Invalid layer shape: {shape}")
        if index == 0:
            topology.append(n_in)
        elif topology[-1] != n_in:
            raise ValueError(f"Invalid topology at layer {index}: expected {topology[-1]}, got {n_in}")
        topology.append(n_out)

    return LayoutInfo(
        layout_path=path,
        mem_dir=path.parent,
        mode=mode,
        num_linear=len(weights),
        topology=topology,
        weight_file=_file_basename(layout.get("weight_all_mem"), "weight_all.mem"),
        bias_file=_file_basename(layout.get("bias_all_mem"), "bias_all.mem"),
        act_file=_file_basename(layout.get("act_all_mem"), "act_all.mem"),
        total_weight_words=int(layout.get("total_weight_words", sum(int(w.get("length", 0)) for w in weights))),
        total_bias_words=int(layout.get("total_bias_words", sum(int(b.get("length", 0)) for b in biases))),
        total_activation_words=int(layout.get("total_activation_words", sum(int(a.get("length", 0)) for a in acts))),
        weights=weights, biases=biases, activations=acts,
    )


def _sv_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().rstrip(",;").replace("_", "")
    if not text:
        return None
    match = re.fullmatch(r"(?:(\d+))?'([sS])?([hHdDbBoO])([0-9a-fA-FxXzZ]+)", text)
    if match:
        base = {"h": 16, "d": 10, "b": 2, "o": 8}[match.group(3).lower()]
        digits = match.group(4).lower().replace("x", "0").replace("z", "0")
        return int(digits, base)
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", text):
        return int(text, 16)
    if re.fullmatch(r"0[bB][01]+", text):
        return int(text, 2)
    return int(text, 10)


def parse_scales_vh(path: str | Path) -> dict[str, int]:
    macros: dict[str, int] = {}
    define_re = re.compile(r"^\s*`define\s+(\w+)\s+([^/\s]+)")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = define_re.match(line)
        if match:
            value = _sv_int(match.group(2))
            if value is not None:
                macros[match.group(1)] = value
    return macros


def validate_scales_against_layout(path: str | Path, info: LayoutInfo, dsp_pw: int) -> dict[str, int]:
    macros = parse_scales_vh(path)
    required = ["NUM_LINEAR", "ACT_W", "WEIGHT_W", "BIAS_W", "DSP_PW"]
    required += [f"N{i}" for i in range(info.num_linear + 1)]
    missing = [key for key in required if key not in macros]
    if missing:
        raise ValueError(f"Missing macros in {path}: {missing}")
    if macros["NUM_LINEAR"] != info.num_linear:
        raise ValueError("NUM_LINEAR disagrees with mem_layout.json")
    for i, expected in enumerate(info.topology):
        if macros[f"N{i}"] != expected:
            raise ValueError(f"N{i} disagrees with mem_layout.json: {macros[f'N{i}']} != {expected}")
    if macros["DSP_PW"] != dsp_pw:
        raise ValueError(
            f"DSP width mismatch: scales.vh has {macros['DSP_PW']}, "
            f"Nexys target requires {dsp_pw}; re-export quantization for Nexys"
        )
    widths = [macros[key] for key in ("ACT_W", "WEIGHT_W", "BIAS_W", "DSP_PW")]
    if min(widths) <= 0 or max(widths[:3]) > dsp_pw:
        raise ValueError("Invalid quantization width contract")
    if macros["ACT_W"] > 16:
        raise ValueError("Current UART transport only supports ACT_W <= 16")
    return macros


def _mem_tokens(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    tokens: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"//.*$", "", line)
        line = re.sub(r"#.*$", "", line)
        tokens.extend(token for token in line.replace(",", " ").split() if token)
    return tokens


def read_mem_words(path: str | Path, width: int) -> list[int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Memory file missing: {path}")
    mask = (1 << width) - 1
    memory: dict[int, int] = {}
    address = 0
    max_address = -1
    for token in _mem_tokens(path):
        if token.startswith("@"):
            address = int(token[1:].replace("_", ""), 16)
            max_address = max(max_address, address - 1)
        else:
            try:
                value = int(token.replace("_", ""), 16)
            except ValueError as exc:
                raise ValueError(f"Invalid readmemh token {token!r} in {path}") from exc
            memory[address] = value & mask
            max_address = max(max_address, address)
            address += 1
    return [memory.get(i, 0) for i in range(max_address + 1)]


def _check_count(name: str, words: list[int], expected: int) -> None:
    if len(words) != expected:
        raise ValueError(f"Wrong {name} memory length: {len(words)} instead of {expected}")


def _format_memory_init(words: list[int], width: int) -> str:
    words = words or [0]
    hex_digits = max(1, (width + 3) // 4)
    return "\n".join(
        f"mem[{i}] = {width}'h{word:0{hex_digits}X};"
        for i, word in enumerate(words)
    )


def _macro(i: int, prefix: str = "N") -> str:
    return f"`{prefix}{i}"


def _sum_expr(parts: Iterable[str]) -> str:
    parts = [p for p in parts if p]
    return " + ".join(parts) if parts else "0"


def _prod_expr(i: int, j: int) -> str:
    return f"({_macro(i)}*{_macro(j)})"


def _case_lines(name: str, entries: list[tuple[int, str]]) -> str:
    return "\n".join(f"{i}: {name} = {value};" for i, value in entries)


def _act_depth_lines(n: int) -> str:
    return "\n".join(
        f"localparam int ACT_DEPTH{i} = ((`ACT{i} == 3) && (`LUT_N{i} > 0)) ? (1 << `LUT_N{i}) : 0;"
        for i in range(n)
    )


def _act_base_cases(n: int) -> list[tuple[int, str]]:
    return [(i, _sum_expr([f"ACT_DEPTH{j}" for j in range(i)])) for i in range(n)]


def build_sequential_sv_variables(info: LayoutInfo, scales: dict[str, int], dsp_pw: int) -> dict[str, dict[str, str]]:
    n = info.num_linear
    a_width, w_width, b_width = (scales[key] for key in ("ACT_W", "WEIGHT_W", "BIAS_W"))
    weights = read_mem_words(info.mem_dir / info.weight_file, w_width)
    biases = read_mem_words(info.mem_dir / info.bias_file, b_width)
    acts = read_mem_words(info.mem_dir / info.act_file, a_width)
    _check_count("weight", weights, info.total_weight_words)
    _check_count("bias", biases, info.total_bias_words)
    _check_count("activation", acts, info.total_activation_words)

    act_depth_localparams = _act_depth_lines(n)
    act_depth_sum_expr = _sum_expr([f"ACT_DEPTH{i}" for i in range(n)])
    ctrl = {
        "layer_size_case": _case_lines("n_at", [(i, _macro(i)) for i in range(n + 1)]),
        "weight_base_case": _case_lines("weight_base_at", [
            (0, "0"),
            *[(i, _sum_expr(_prod_expr(j, j + 1) for j in range(i))) for i in range(1, n)],
        ]),
        "bias_base_case": _case_lines("bias_base_at", [
            (0, "0"),
            *[(i, _sum_expr(_macro(j) for j in range(1, i + 1))) for i in range(1, n)],
        ]),
    }
    datapath = {
        "act_depth_localparams": "\n".join([
            act_depth_localparams,
            f"localparam int ACT_DEPTH_RAW = {act_depth_sum_expr};",
            "localparam int ACT_DEPTH = (ACT_DEPTH_RAW > 0) ? ACT_DEPTH_RAW : 1;",
        ]),
        "sx_case": _case_lines("sx_at", [(i, _macro(i, "SX")) for i in range(n + 1)]),
        "sw_case": _case_lines("sw_at", [(i, _macro(i, "SW")) for i in range(n)]),
        "sb_case": _case_lines("sb_at", [(i, _macro(i, "SB")) for i in range(n)]),
        "sp_case": _case_lines("sp_at", [(i, _macro(i, "SP")) for i in range(max(0, n - 1))]),
        "act_case": _case_lines("act_at", [(i, _macro(i, "ACT")) for i in range(n)]),
        "lutn_case": _case_lines("lutn_at", [(i, _macro(i, "LUT_N")) for i in range(n)]),
        "act_base_case": _case_lines("act_base_at", _act_base_cases(n)),
    }
    common = {
        "dsp_pw_expr": str(dsp_pw),
        "output_size_expr": _macro(n),
        "max_neurons_expr": str(max(info.topology)),
    }
    return {
        "common": common,
        "rom_weight_all.sv": {"memory_weight": _format_memory_init(weights, w_width)},
        "rom_bias_all.sv": {"memory_bias": _format_memory_init(biases, b_width)},
        "rom_act_all.sv": {"memory_act_function": _format_memory_init(acts, a_width)},
        "mlp_sequential_ctrl.sv": ctrl,
        "mlp_sequential_datapath.sv": datapath,
        "mlp_sequential.sv": {
            "weight_depth_expr": _sum_expr(_prod_expr(i, i + 1) for i in range(n)),
            "bias_depth_expr": _sum_expr(_macro(i) for i in range(1, n + 1)),
            "act_depth_localparams": act_depth_localparams,
            "act_depth_sum_expr": act_depth_sum_expr,
        },
        "soc_mlp.sv": common,
    }


def render_sv_placeholders(text: str, variables: dict[str, str], *, filename: str) -> str:
    """Replace named placeholders only; preserve actual SV braces unchanged."""
    rendered = text
    for name, value in variables.items():
        line_pattern = re.compile(rf"^(?P<indent>[ \t]*)\{{{re.escape(name)}\}}[ \t]*$", re.M)

        def replacement(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return "\n".join(indent + line if line else "" for line in value.splitlines())

        rendered = line_pattern.sub(replacement, rendered)
    for name, value in variables.items():
        rendered = rendered.replace("{" + name + "}", value)
    remaining = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", rendered)) & KNOWN_PLACEHOLDERS
    if remaining:
        raise ValueError(f"Unresolved placeholders in {filename}: {sorted(remaining)}")
    return rendered


def sync_scales_vh(template_dir: Path, out_dir: Path, mem_dir: Path) -> Path:
    source = mem_dir / "scales.vh"
    if not source.is_file():
        source = template_dir / "scales.vh"
    if not source.is_file():
        raise FileNotFoundError("scales.vh not found in the export or Nexys template")
    target = out_dir / "scales.vh"
    shutil.copy2(source, target)
    return target


def copy_and_render_nexys_rtl(
    repo_root: str | Path,
    info: LayoutInfo,
    *,
    template_rel: str,
    generated_rel: str,
    clean_generated: bool,
    dsp_pw: int,
) -> Path:
    root = Path(repo_root).resolve()
    template_dir = (root / template_rel).resolve()
    out_dir = (root / generated_rel).resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Nexys template directory not found: {template_dir}")
    if template_dir == out_dir or template_dir in out_dir.parents:
        raise ValueError("Generated directory must not overwrite or contain the templates")
    missing = [name for name in NEXYS_RTL_FILES_SEQUENTIAL if not (template_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Nexys RTL templates: {missing}")
    if clean_generated and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = sync_scales_vh(template_dir, out_dir, info.mem_dir)
    scales = validate_scales_against_layout(header, info, dsp_pw)
    per_file = build_sequential_sv_variables(info, scales, dsp_pw)
    common = per_file.get("common", {})
    # Render mandatory sources and optional extra SystemVerilog helpers.
    for source in sorted(template_dir.glob("*.sv")):
        substitutions = {**common, **per_file.get(source.name, {})}
        rendered = render_sv_placeholders(
            source.read_text(encoding="utf-8"), substitutions, filename=source.name
        )
        (out_dir / source.name).write_text(rendered, encoding="utf-8")
    return out_dir


# Pipeline export helpers are kept local so the Nexys backend does not
# depend on Lattice tooling or its 32-bit DSP configuration.
def _resolve_mem_path(mem_dir: Path, path_like: Any, fallback: str) -> Path:
    if path_like is None:
        return mem_dir / fallback
    candidate = Path(str(path_like))
    if candidate.is_absolute():
        return candidate if candidate.exists() else mem_dir / candidate.name
    return candidate if candidate.exists() else (
        mem_dir / candidate if (mem_dir / candidate).exists() else mem_dir / candidate.name
    )


def _check_word_count(kind: str, mem_path: Path, words: list[int], expected: int, *, allow_empty_as_zero: bool = False) -> None:
    if not (allow_empty_as_zero and len(words) == 0 and expected == 0):
        _check_count(kind, words, expected)


def _numeric_widths(scales: dict[str, int], dsp_pw: int) -> tuple[int, int, int]:
    result = tuple(int(scales[x]) for x in ("ACT_W", "WEIGHT_W", "BIAS_W"))
    if int(scales["DSP_PW"]) != dsp_pw or max(result) > dsp_pw:
        raise ValueError("Pipeline scales mismatch with DSP width")
    return result


def _csv_expr(items: Iterable[str]) -> str:
    return ", ".join(items)


def _macro_list(prefix: str, count: int, *, start: int = 0) -> str:
    return _csv_expr(f"`{prefix}{i}" for i in range(start, start + count))


def _pipeline_buffer_name(idx: int, signal: str) -> str:
    return f"inter{idx}_{signal}"


def build_intermediate_buffers(info: LayoutInfo) -> str:
    lines: list[str] = []
    n = info.num_linear
    for b in range(max(0, n - 1)):
        depth_macro = _macro(b + 1)
        aw_name = f"INTER{b}_AW"
        lines.extend(
            [
                f"localparam int INTER{b}_DEPTH = {depth_macro};",
                f"localparam int {aw_name} = (INTER{b}_DEPTH > 1) ? $clog2(INTER{b}_DEPTH) : 1;",
                "",
                f"logic                 inter{b}_rd_en;",
                f"logic [{aw_name}-1:0] inter{b}_rd_addr;",
                f"logic signed [ACT_W-1:0] inter{b}_rd_data;",
                "",
                f"logic                 inter{b}_wr_en;",
                f"logic [{aw_name}-1:0] inter{b}_wr_addr;",
                f"logic signed [ACT_W-1:0] inter{b}_wr_data;",
                "",
                "pp_buffer #(",
                "    .ACT_W(ACT_W),",
                f"    .DEPTH(INTER{b}_DEPTH),",
                f"    .AW   ({aw_name})",
                f") u_inter_pp_{b} (",
                "    .clk        (clk),",
                "",
                "    .rd_bank_sel(phase_sel),",
                f"    .rd_en      (inter{b}_rd_en),",
                f"    .rd_addr    (inter{b}_rd_addr),",
                f"    .rd_data    (inter{b}_rd_data),",
                "",
                "    .wr_bank_sel(~phase_sel),",
                f"    .wr_en      (inter{b}_wr_en),",
                f"    .wr_addr    (inter{b}_wr_addr),",
                f"    .wr_data    (inter{b}_wr_data)",
                ");",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _layer_input_ports(layer_idx: int, num_layers: int) -> dict[str, str]:
    if num_layers == 1 or layer_idx == 0:
        return {
            "in_rd_en": "in_rd_en",
            "in_rd_addr": "in_rd_addr",
            "in_rd_data": "in_rd_data",
        }
    prev = layer_idx - 1
    return {
        "in_rd_en": f"inter{prev}_rd_en",
        "in_rd_addr": f"inter{prev}_rd_addr",
        "in_rd_data": f"inter{prev}_rd_data",
    }


def _layer_output_ports(layer_idx: int, num_layers: int) -> dict[str, str]:
    if num_layers == 1 or layer_idx == num_layers - 1:
        return {
            "out_wr_en": "out_wr_en",
            "out_wr_addr": "out_wr_addr",
            "out_wr_data": "out_wr_data",
        }
    return {
        "out_wr_en": f"inter{layer_idx}_wr_en",
        "out_wr_addr": f"inter{layer_idx}_wr_addr",
        "out_wr_data": f"inter{layer_idx}_wr_data",
    }


def build_layer_instances(info: LayoutInfo) -> str:
    lines: list[str] = []
    n = info.num_linear
    for l in range(n):
        p_in = _layer_input_ports(l, n)
        p_out = _layer_output_ports(l, n)
        has_sp = "1'b1" if l < n - 1 else "1'b0"
        sp_expr = _macro(l, "SP") if l < n - 1 else _macro(l + 1, "SX")

        lines.extend(
            [
                f"layer_{l} #(",
                f"    .ACT_TYPE         ({_macro(l, 'ACT')}),",
                f"    .N_IN             ({_macro(l)}),",
                f"    .N_OUT            ({_macro(l + 1)}),",
                "    .ACT_W            (ACT_W),",
                "    .WEIGHT_W         (WEIGHT_W),",
                "    .BIAS_W           (BIAS_W),",
                "    .LEAK_SHIFT       (LEAK_SHIFT_CFG),",
                f"    .HAS_SP           ({has_sp}),",
                f"    .Sx               ({_macro(l, 'SX')}),",
                f"    .Sw               ({_macro(l, 'SW')}),",
                f"    .Sb               ({_macro(l, 'SB')}),",
                f"    .Sp               ({sp_expr}),",
                f"    .Sx_out           ({_macro(l + 1, 'SX')}),",
                f"    .LUT_N            ({_macro(l, 'LUT_N')}),",
                "    .DSP_PW           (DSP_PW)",
                f") u_layer_{l} (",
                "    .clk        (clk),",
                "    .rstn       (rstn),",
                "    .start      (start_phase_pulse),",
                f"    .in_rd_en   ({p_in['in_rd_en']}),",
                f"    .in_rd_addr ({p_in['in_rd_addr']}),",
                f"    .in_rd_data ({p_in['in_rd_data']}),",
                f"    .out_wr_en  ({p_out['out_wr_en']}),",
                f"    .out_wr_addr({p_out['out_wr_addr']}),",
                f"    .out_wr_data({p_out['out_wr_data']}),",
                f"    .done       (layer_done_pulse[{l}])",
                ");",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def build_pipeline_sv_variables(info: LayoutInfo, scales: dict[str, int], *, dsp_pw: int = 32) -> dict[str, dict[str, str]]:
    """Build placeholder dictionaries for the pipeline templates."""
    n = info.num_linear
    act_width, w_width, b_width = _numeric_widths(scales, dsp_pw)

    common = {
        "dsp_pw_expr": str(int(dsp_pw)),
        "output_size_expr": _macro(n),
        "max_neurons_expr": str(int(max(info.topology))),
    }

    mlp_vars = {
        "n_tab_expr": _macro_list("N", n + 1),
        "sx_tab_expr": _macro_list("SX", n + 1),
        "sw_tab_expr": _macro_list("SW", n),
        "sb_tab_expr": _macro_list("SB", n),
        "sp_tab_expr": _macro_list("SP", max(0, n - 1)) if n > 1 else "0",
        "act_tab_expr": _macro_list("ACT", n),
        "lutn_tab_expr": _macro_list("LUT_N", n),
        "intermediate_buffers": build_intermediate_buffers(info),
        "layer_instances": build_layer_instances(info),
    }

    per_file: dict[str, dict[str, str]] = {
        "common": common,
        "mlp_pipeline.sv": mlp_vars,
        "soc_mlp.sv": common,
    }

    bias_by_idx = {int(b.get("layer_idx", i)): b for i, b in enumerate(info.biases)}
    act_by_idx = {int(a.get("layer_idx", i)): a for i, a in enumerate(info.activations)}

    for i, w_item in enumerate(info.weights):
        layer_idx = int(w_item.get("layer_idx", i))
        b_item = bias_by_idx.get(layer_idx)
        if b_item is None:
            raise ValueError(f"Missing bias for layer_idx={layer_idx}")

        w_path = _resolve_mem_path(info.mem_dir, w_item.get("file"), f"weight_layer{layer_idx}.mem")
        b_path = _resolve_mem_path(info.mem_dir, b_item.get("file"), f"bias_layer{layer_idx}.mem")

        w_words = read_mem_words(w_path, w_width)
        b_words = read_mem_words(b_path, b_width)

        _check_word_count(f"weight layer {layer_idx}", w_path, w_words, int(w_item.get("length", len(w_words))))
        _check_word_count(f"bias layer {layer_idx}", b_path, b_words, int(b_item.get("length", len(b_words))))

        layer_name = f"layer_{layer_idx}"
        weight_name = f"rom_weight_{layer_idx}"
        bias_name = f"rom_bias_{layer_idx}"
        act_name = f"rom_act_{layer_idx}"

        per_file[f"layer_{layer_idx}.sv"] = {
            "module_layer_name": layer_name,
            "module_weight_name": weight_name,
            "module_bias_name": bias_name,
            "module_act_name": act_name,
        }
        per_file[f"rom_weight_{layer_idx}.sv"] = {
            "module_weight_name": weight_name,
            "memory_weight": _format_memory_init(w_words, w_width),
        }
        per_file[f"rom_bias_{layer_idx}.sv"] = {
            "module_bias_name": bias_name,
            "memory_bias": _format_memory_init(b_words, b_width),
        }

        a_item = act_by_idx.get(layer_idx)
        act_len = int(a_item.get("length", 0)) if a_item is not None else 0
        act_file = a_item.get("file") if a_item is not None else None
        if act_len > 0:
            a_path = _resolve_mem_path(info.mem_dir, act_file, f"act_layer{layer_idx}.mem")
            a_words = read_mem_words(a_path, act_width)
            _check_word_count(f"activation layer {layer_idx}", a_path, a_words, act_len, allow_empty_as_zero=True)
            per_file[f"rom_act_{layer_idx}.sv"] = {
                "module_act_name": act_name,
                "memory_act_function": _format_memory_init(a_words, act_width),
            }

    return per_file



def copy_and_render_nexys_pipeline_rtl(
    repo_root: str | Path,
    info: LayoutInfo,
    *,
    template_rel: str,
    generated_rel: str,
    clean_generated: bool,
    dsp_pw: int,
) -> Path:
    root = Path(repo_root).resolve()
    template_dir = (root / template_rel).resolve()
    out_dir = (root / generated_rel).resolve()
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Nexys pipeline templates missing: {template_dir}")
    if template_dir == out_dir or template_dir in out_dir.parents:
        raise ValueError("Generated directory cannot overwrite templates")
    required = NEXYS_RTL_FILES_PIPELINE_STATIC + list(NEXYS_PIPELINE_PER_LAYER_TEMPLATES)
    missing = [name for name in required if not (template_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Nexys pipeline templates: {missing}")
    if clean_generated and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = sync_scales_vh(template_dir, out_dir, info.mem_dir)
    scales = validate_scales_against_layout(header, info, dsp_pw)
    substitutions = build_pipeline_sv_variables(info, scales, dsp_pw=dsp_pw)
    common = substitutions.get("common", {})
    for name in NEXYS_RTL_FILES_PIPELINE_STATIC:
        rendered = render_sv_placeholders(
            (template_dir / name).read_text(encoding="utf-8"),
            {**common, **substitutions.get(name, {})},
            filename=name,
        )
        (out_dir / name).write_text(rendered, encoding="utf-8")
    for i in range(info.num_linear):
        for name, template in (
            (f"layer_{i}.sv", "layer.sv"),
            (f"rom_weight_{i}.sv", "rom_weight.sv"),
            (f"rom_bias_{i}.sv", "rom_bias.sv"),
        ):
            rendered = render_sv_placeholders(
                (template_dir / template).read_text(encoding="utf-8"),
                {**common, **substitutions[name]},
                filename=name,
            )
            (out_dir / name).write_text(rendered, encoding="utf-8")
        name = f"rom_act_{i}.sv"
        if name in substitutions:
            rendered = render_sv_placeholders(
                (template_dir / "rom_act.sv").read_text(encoding="utf-8"),
                {**common, **substitutions[name]},
                filename=name,
            )
            (out_dir / name).write_text(rendered, encoding="utf-8")
    return out_dir


def find_nexys_xdc(root: Path, template_dir: Path) -> Path:
    candidates = [
        root / "constraints" / "nexys_a7.xdc",
        template_dir / "constraints" / "nexys_a7.xdc",
        template_dir / "contraints" / "nexys_a7.xdc",
        template_dir / "nexys_a7.xdc",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Missing nexys_a7.xdc. Put your verified pin constraints in "
        f"{candidates[0]} (or inside the Nexys RTL template)."
    )


def _tcl_quote(value: str | Path) -> str:
    """Use TCL braced literal for filesystem paths (protect spaces)."""
    s = str(value).replace("\\", "/")
    if any(char in s for char in "{}\n\r"):
        raise ValueError(f"Unsupported Tcl path/value: {value!r}")
    return "{" + s + "}"


def make_nexys_tcl(
    repo_root: str | Path,
    info: LayoutInfo,
    rtl_dir: Path,
    *,
    xdc_path: Path,
    project_name: str,
    part_name: str,
    jobs: int,
    top_name: str = "SOC_MLP",
    tcl_path: str | Path | None = None,
) -> Path:
    root = Path(repo_root).resolve()
    rtl_dir = Path(rtl_dir).resolve()
    xdc_path = Path(xdc_path).resolve()
    tcl_name = ("create_soc_mlp_project_nexys_pipeline.tcl" if info.mode == "pipeline" else "create_soc_mlp_project_nexys.tcl")
    output = Path(tcl_path or root / "vivado" / tcl_name).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(path for path in rtl_dir.glob("*.sv") if path.is_file())
    if not sources:
        raise ValueError(f"No generated SystemVerilog files in {rtl_dir}")
    header = rtl_dir / "scales.vh"
    if not header.is_file():
        raise FileNotFoundError(f"Missing generated scales.vh: {header}")
    source_lines = "\n".join("    " + _tcl_quote(p) for p in sources)
    tcl_text = f'''# Auto-generated Nexys A7 Artix-7 Vivado project (UART MLP)
# Topology: {" -> ".join(map(str, info.topology))}
# Architecture: {info.mode}
set project_name {_tcl_quote(project_name)}
set part_name {_tcl_quote(part_name)}
set top_name {_tcl_quote(top_name)}
set project_dir {_tcl_quote(root / 'vivado' / 'build' / project_name)}
set rtl_dir {_tcl_quote(rtl_dir)}
set xdc_path {_tcl_quote(xdc_path)}
set header_path {_tcl_quote(header)}
set run_flow 0
foreach arg $argv {{
    if {{$arg eq "--run"}} {{set run_flow 1}}
}}

# Never silently delete an existing Vivado project.
if {{[file exists [file join $project_dir "$project_name.xpr"]]}} {{
    error "Project already exists: $project_dir. Rename/move it or use a new project_name."
}}
file mkdir $project_dir
create_project $project_name $project_dir -part $part_name
set_property target_language Verilog [current_project]
set_property include_dirs [list $rtl_dir] [get_filesets sources_1]

set rtl_files {{
{source_lines}
}}
foreach src $rtl_files {{
    if {{![file isfile $src]}} {{error "Missing RTL source: $src"}}
    add_files -norecurse -fileset sources_1 $src
    set_property file_type SystemVerilog [get_files $src]
}}
add_files -norecurse -fileset sources_1 $header_path
set_property file_type {{Verilog Header}} [get_files $header_path]
if {{![file isfile $xdc_path]}} {{error "Missing XDC file: $xdc_path"}}
add_files -norecurse -fileset constrs_1 $xdc_path
set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "INFO: Nexys A7 Vivado project created: $project_dir"

if {{$run_flow}} {{
    launch_runs synth_1 -jobs {int(jobs)}
    wait_on_run synth_1
    if {{[get_property PROGRESS [get_runs synth_1]] ne "100%"}} {{
        error "Synthesis did not complete successfully"
    }}
    if {{[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"}} {{
        error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"
    }}
    launch_runs impl_1 -to_step write_bitstream -jobs {int(jobs)}
    wait_on_run impl_1
    if {{[get_property PROGRESS [get_runs impl_1]] ne "100%"}} {{
        error "Implementation/bitstream did not complete successfully"
    }}
}}
'''
    output.write_text(tcl_text, encoding="utf-8")
    return output


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
    """Render independent Nexys RTL and emit a complete Vivado Tcl file."""
    root = Path(repo_root).resolve()
    info = parse_layout(mem_layout_path, mode_override=fpga_mode)
    if jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if dsp_pw < 1:
        raise ValueError("dsp_pw must be a positive integer")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", project_name):
        raise ValueError("Invalid Vivado project name")
    template_dir = (root / template_rel).resolve()
    # Check constraints before generating output, to fail early.
    xdc_path = find_nexys_xdc(root, template_dir)
    renderer = copy_and_render_nexys_pipeline_rtl if info.mode == 'pipeline' else copy_and_render_nexys_rtl
    rtl_dir = renderer(
        root, info,
        template_rel=template_rel,
        generated_rel=generated_rel,
        clean_generated=clean_generated,
        dsp_pw=dsp_pw,
    )
    tcl_path = make_nexys_tcl(
        root, info, rtl_dir,
        xdc_path=xdc_path,
        project_name=project_name,
        part_name=part_name,
        jobs=jobs,
    )
    return {
        "mode": info.mode,
        "topology": info.topology,
        "num_linear": info.num_linear,
        "mem_dir": str(info.mem_dir),
        "layout_path": str(info.layout_path),
        "generated_rtl_dir": str(rtl_dir),
        "generated_tcl": str(tcl_path),
        "project_name": project_name,
        "device": part_name,
        "top_name": "SOC_MLP",
        "dsp_pw": dsp_pw,
        "generated_roms": (
            {"weight": [str(rtl_dir / f"rom_weight_{i}.sv") for i in range(info.num_linear)],
             "bias": [str(rtl_dir / f"rom_bias_{i}.sv") for i in range(info.num_linear)],
             "layers": [str(rtl_dir / f"layer_{i}.sv") for i in range(info.num_linear)]}
            if info.mode == "pipeline" else
            {"weight": str(rtl_dir / "rom_weight_all.sv"),
             "bias": str(rtl_dir / "rom_bias_all.sv"),
             "activation": str(rtl_dir / "rom_act_all.sv")}
        ),
        "total_weight_words": info.total_weight_words,
        "total_bias_words": info.total_bias_words,
        "total_activation_words": info.total_activation_words,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate standalone Nexys A7 sequential RTL and Vivado Tcl"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--fpga-mode", default="sequential", choices=["sequential", "pipeline"])
    parser.add_argument("--template-rel", default=None)
    parser.add_argument("--generated-rel", default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--part-name", default="xc7a100tcsg324-1")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--dsp-pw", type=int, default=48)
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    layout = Path(args.layout)
    if not layout.is_absolute():
        layout = root / layout
    result = generate_project(
        root, layout,
        fpga_mode=args.fpga_mode,
        template_rel=args.template_rel or f"src/rtl/src_nexys_{args.fpga_mode}",
        generated_rel=args.generated_rel or f"src/rtl/generated/src_nexys_{args.fpga_mode}",
        project_name=args.project_name or f"SOC_MLP_nexys_a7_{args.fpga_mode}",
        part_name=args.part_name,
        jobs=args.jobs,
        dsp_pw=args.dsp_pw,
        clean_generated=not args.no_clean,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
