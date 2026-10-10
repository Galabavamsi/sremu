"""WP12: docs/model_card.yaml is the single source of paper Table II (DESIGN 5, 5.15).

Oracle: the validity-domain table of DESIGN.md section 5.15, parsed from the spec itself.
The YAML must reproduce every row (component, model, validity domain, evidence ids after
range expansion, default) in order; the generated Markdown must be up to date; the LaTeX
rendering must be ASCII-only and escape TeX specials.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
DESIGN = REPO / "docs" / "DESIGN.md"
CARD = REPO / "docs" / "model_card.yaml"
GEN = REPO / "docs" / "tools" / "gen_model_card.py"

pytestmark = pytest.mark.unit


def _load_gen() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_model_card", GEN)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_model_card"] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_gen()


# --------------------------------------------------------------------------- spec oracle
def expand_ids(cell: str) -> list[str]:
    """'PH-02–07, G1–G3, V-TANG' -> ['PH-02', ..., 'PH-07', 'G1', 'G2', 'G3', 'V-TANG']."""
    out: list[str] = []
    prefix, width = "", 2
    for tok in (t.strip() for t in cell.split(",")):
        m = re.fullmatch(r"([A-Z]+(?:-[A-Z]+)*-?)(\d+)(?:[–-]([A-Z]+-?)?(\d+))?", tok)
        if m:
            prefix, first = m.group(1), m.group(2)
            width = len(first)
            last = m.group(4) or first
            out += [f"{prefix}{str(i).zfill(width)}" for i in range(int(first), int(last) + 1)]
        elif re.fullmatch(r"\d+", tok):  # bare number inherits the previous prefix
            out.append(f"{prefix}{tok.zfill(width)}")
        else:
            out.append(tok)
    return out


def design_rows() -> list[dict[str, object]]:
    text = DESIGN.read_text(encoding="utf-8")
    start = text.index("### 5.15 Validity domain")
    lines = text[start:].splitlines()[1:]
    table = [
        ln
        for ln in lines[: next(i for i, ln in enumerate(lines) if ln.startswith("---"))]
        if ln.startswith("|")
    ]
    header = [c.strip() for c in table[0].strip("|").split("|")]
    assert header == ["Component", "Model", "Validity domain", "Evidence", "Default"]
    rows = []
    for ln in table[2:]:
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        rows.append(
            {
                "component": c[0],
                "model": c[1],
                "validity_domain": c[2],
                "evidence": expand_ids(c[3]),
                "default": c[4],
            }
        )
    return rows


def _norm(s: str) -> str:
    return " ".join(s.replace("`", "").split())


def test_expand_ids_teeth() -> None:
    assert expand_ids("PH-02–07, G1–G3, V-TANG") == [
        "PH-02",
        "PH-03",
        "PH-04",
        "PH-05",
        "PH-06",
        "PH-07",
        "G1",
        "G2",
        "G3",
        "V-TANG",
    ]
    assert expand_ids("PH-13–15, 25, 30") == ["PH-13", "PH-14", "PH-15", "PH-25", "PH-30"]
    assert expand_ids("PH-10, PH-32, G8") == ["PH-10", "PH-32", "G8"]


def test_model_card_rows_match_design_5_15() -> None:
    spec = design_rows()
    card = gen.load(CARD)["rows"]
    assert len(spec) == 13
    assert len(card) == len(spec)
    for s, c in zip(spec, card, strict=True):
        for key in ("component", "model", "validity_domain", "default"):
            assert _norm(c[key]) == _norm(str(s[key])), (c["id"], key)
        assert c["evidence"] == s["evidence"], c["id"]


def test_generated_markdown_is_up_to_date() -> None:
    doc = gen.load(CARD)
    assert (REPO / "docs" / "model_card.md").read_text(encoding="utf-8") == gen.render_md(doc)


def test_latex_table_is_ascii_escaped_and_complete() -> None:
    doc = gen.load(CARD)
    tex = gen.render_tex(doc)
    assert all(ord(ch) < 128 for ch in tex)
    assert r"\label{tab:modelcard}" in tex
    body = tex[tex.index(r"\midrule") : tex.index(r"\bottomrule")]
    assert sum(1 for ln in body.splitlines() if ln.rstrip().endswith("\\\\")) == 13
    assert r"\texttt{po\_patch}" in tex
    no_math = re.sub(r"\$[^$]*\$", "", tex)  # an underscore outside math must be escaped
    assert not re.search(r"(?<!\\)_", no_math.split(r"\midrule", 1)[1])


def _quantities(s: str) -> list[float]:
    """Numeric values in a cell (sign dropped); TeX 10^{-3} and plain 1e-3 agree."""
    s = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", s)
    return sorted(float(x) for x in re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", s))


# The one override that rewrites the spec text on purpose: "Pipeline (§5.6)" points to a
# DESIGN section the paper does not have, so the LaTeX cell spells the §5.6 order out.
_REWORDED = {("nonidealities", "model")}


def test_tex_overrides_keep_the_spec_quantities_and_symbols() -> None:
    """A *_tex override is typed by hand; it must carry the same numbers and the same
    Greek letters / relations as the verbatim cell it replaces (e.g. 1e-3 vs 10^{-3})."""
    card = gen.load(CARD)["rows"]
    n_overrides = 0
    for r in card:
        for key in ("component", "model", "validity_domain", "default"):
            if f"{key}_tex" not in r or (r["id"], key) in _REWORDED:
                continue
            n_overrides += 1
            plain, tex = r[key], r[f"{key}_tex"]
            assert _quantities(plain) == _quantities(tex), (r["id"], key)
            for ch, cmd in gen.UNICODE_TEX.items():
                if ch in plain and cmd.startswith("$\\"):
                    assert cmd.strip("$") in tex, (r["id"], key, ch)
    assert n_overrides == 8
    nonideal = next(r for r in card if r["id"] == "nonidealities")
    order = ["command", "quantizer", "failures", "table", "efficiency", "coupling", "transition"]
    pos = [nonideal["model_tex"].lower().index(w) for w in order]
    assert pos == sorted(pos)  # DESIGN 5.6 normative order


def test_quantities_helper_teeth() -> None:
    assert _quantities("ε_rel ≤ 1e-3") == _quantities(r"$\varepsilon \leq 10^{-3}$") == [1e-3]
    assert _quantities("B ≤ 0.8 f_s; ≤ −60 dB") == [0.8, 60.0]
    assert _quantities(r"$10^{-2}$") != _quantities("1e-3")


def test_tex_escape_maps_unicode_and_specials() -> None:
    assert gen.tex_escape("r ≥ 3λ & 50%") == r"r $\geq$ 3$\lambda$ \& 50\%"
    assert gen.tex_escape("`a_b`") == r"\texttt{a\_b}"
    with pytest.raises(gen.ModelCardError):
        gen.tex_escape("unmapped ∞")


def test_compress_ids() -> None:
    assert gen.compress_ids(["PH-02", "PH-03", "PH-04"]) == ["PH-02--04"]
    assert gen.compress_ids(["PH-13", "PH-14", "PH-15", "PH-25", "PH-30"]) == [
        "PH-13--15",
        "PH-25",
        "PH-30",
    ]
    assert gen.compress_ids(["PH-28", "PH-29"]) == ["PH-28", "PH-29"]  # two: not a range
    assert gen.compress_ids(["G1", "G2", "G3", "V-TANG"]) == ["G1--3", "V-TANG"]


def test_validate_rejects_yaml_boolean_and_bad_ids() -> None:
    doc = yaml.safe_load(CARD.read_text(encoding="utf-8"))
    doc["rows"][0]["default"] = yaml.safe_load("On")  # unquoted On is a YAML 1.1 boolean
    with pytest.raises(gen.ModelCardError, match="must be a string"):
        gen.validate(doc)
    doc = yaml.safe_load(CARD.read_text(encoding="utf-8"))
    doc["rows"][0]["evidence"] = ["PH-1"]
    with pytest.raises(gen.ModelCardError, match="not a spec test id"):
        gen.validate(doc)


def test_cli_check_detects_stale_markdown(tmp_path: Path) -> None:
    md = tmp_path / "model_card.md"
    assert gen.main(["--md", str(md)]) == 0
    assert gen.main(["--check", "--md", str(md)]) == 0
    md.write_text(md.read_text(encoding="utf-8") + "edited\n", encoding="utf-8")
    assert gen.main(["--check", "--md", str(md)]) == 1
