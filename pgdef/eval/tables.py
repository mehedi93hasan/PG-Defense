"""Manuscript table generators (Tables II-IX).

Each function consumes a results structure produced by the evaluation scripts
and emits the corresponding table both as a console-friendly ``pandas`` frame
and as IEEEtran/booktabs LaTeX matching the manuscript layout. Only the outputs
that appear in the manuscript are produced -- no auxiliary tables.

All numeric inputs are computed by the pipeline; nothing is hard-coded, so the
tables reflect whatever data and hardware the scripts were run on.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

DATASETS = ["CICIDS2017", "UNSW-NB15", "Edge-IIoTset"]
METHODS_ORDER = ["FA-CNN", "GTAE-IDS", "DAE", "Adv. Retrain", "DNN",
                 "Std. RF", "PG-Def"]


def _pct(x: float, nd: int = 1) -> str:
    return "--" if x is None or pd.isna(x) else f"{100 * x:.{nd}f}"


def _latex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=False, column_format="l" + "r" * (df.shape[1] - 1))
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n{body}\\end{{table}}\n")


# --- Table II: clean-data detection performance + memory ------------------- #
def table_clean(results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, object]:
    """results[method][dataset] = {tpr, fpr, ..., ram_mb}."""
    rows = []
    for m in METHODS_ORDER:
        if m not in results:
            continue
        r = results[m]
        row = {"Method": m}
        for ds in DATASETS:
            d = r.get(ds, {})
            row[f"{ds} TPR"] = _pct(d.get("tpr"))
            row[f"{ds} FPR"] = _pct(d.get("fpr"))
        row["Mem (MB)"] = f"{r.get('memory_mb', float('nan')):.1f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    return {"df": df,
            "latex": _latex(df, "Clean-data detection performance and memory "
                                "footprint across three benchmark datasets",
                            "tab:clean")}


# --- Table III: per-class TPR ---------------------------------------------- #
def table_per_class(per_class: Dict[str, Dict[str, Dict[str, float]]],
                    classes: List[str],
                    methods=("FA-CNN", "GTAE-IDS", "Adv. Retrain", "PG-Def")
                    ) -> Dict[str, object]:
    """per_class[dataset][method][class] = tpr."""
    frames = {}
    for ds in ["CICIDS2017", "UNSW-NB15"]:
        rows = []
        for cls in classes:
            row = {"Attack Class": cls}
            for m in methods:
                row[m] = _pct(per_class.get(ds, {}).get(m, {}).get(cls))
            rows.append(row)
        frames[ds] = pd.DataFrame(rows)
    return {"df": frames,
            "latex": "\n".join(
                _latex(f, f"Per-class TPR (\\%) on {ds}", f"tab:perclass_{ds}")
                for ds, f in frames.items())}


# --- Table IV: resource consumption ---------------------------------------- #
def table_resources(profiles: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """profiles[method] = {ram_mb, latency_ms, cpu, power_w, throughput_fps}."""
    rows = []
    for m in METHODS_ORDER:
        if m not in profiles:
            continue
        p = profiles[m]
        rows.append({
            "Method": m,
            "RAM (MB)": f"{p.get('ram_mb', float('nan')):.1f}",
            "Lat (ms)": f"{p.get('latency_ms', float('nan')):.2f}",
            "CPU (%)": f"{p.get('cpu', float('nan')):.1f}" if p.get('cpu') else "--",
            "Pwr (W)": f"{p.get('power_w'):.1f}" if p.get('power_w') else "--",
            "Tput (fl/s)": f"{p.get('throughput_fps', float('nan')):.0f}",
        })
    df = pd.DataFrame(rows)
    return {"df": df,
            "latex": _latex(df, "Resource consumption comparison across platforms",
                            "tab:resources")}


# --- Table V: ablation ----------------------------------------------------- #
def table_ablation(ablation: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """ablation[config] = {k, CICIDS2017, UNSW-NB15, Edge-IIoTset}."""
    rows = []
    for cfg, vals in ablation.items():
        row = {"Configuration": cfg, "k": vals.get("k", "")}
        for ds in DATASETS:
            row[ds] = _pct(vals.get(ds))
        rows.append(row)
    df = pd.DataFrame(rows)
    return {"df": df,
            "latex": _latex(df, "Ablation study: adversarial TPR (\\%) by "
                                "feature category removal and group contribution "
                                "(vs. SAAE)", "tab:ablation")}


# --- Table VI: cross-domain matrix ----------------------------------------- #
def table_cross_matrix(matrix: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """matrix[train][test] = tpr."""
    rows = []
    for tr in DATASETS:
        row = {"Train -> Test": tr}
        for te in DATASETS:
            row[te] = _pct(matrix.get(tr, {}).get(te))
        rows.append(row)
    df = pd.DataFrame(rows)
    return {"df": df,
            "latex": _latex(df, "PG-Def cross-domain generalisation matrix (TPR \\%)",
                            "tab:crossmatrix")}


# --- Table VII: cross-domain summary --------------------------------------- #
def table_cross_summary(summary: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """summary[method] = {same, cross}."""
    rows = []
    for m in METHODS_ORDER:
        if m not in summary:
            continue
        s = summary[m]
        same, cross = s["same"], s["cross"]
        rows.append({
            "Method": m,
            "Same": _pct(same), "Cross": _pct(cross),
            "$\\Delta$TPR": f"{100 * (cross - same):+.1f}",
            "Rel. Drop": f"{100 * (same - cross) / same:.1f}\\%" if same else "--",
        })
    df = pd.DataFrame(rows)
    return {"df": df,
            "latex": _latex(df, "Cross-domain generalisation: average "
                                "off-diagonal TPR (\\%)", "tab:crosssummary")}


# --- Table VIII: adversarial robustness ------------------------------------ #
def table_adversarial(adv: Dict[str, Dict[str, Dict[str, float]]]
                      ) -> Dict[str, object]:
    """adv[dataset][method][attack] = tpr_adv (attacks incl. SAAE)."""
    cols = ["FGSM", "BIM", "CW2", "CWinf", "DF", "JSMA", "SAAE"]
    ds_short = {"CICIDS2017": "CIC", "UNSW-NB15": "UNSW", "Edge-IIoTset": "Edge"}
    methods = ["FA-CNN", "GTAE-IDS", "Adv. Retrain", "PG-Def"]
    rows = []
    for ds in DATASETS:
        for m in methods:
            d = adv.get(ds, {}).get(m, {})
            if not d:
                continue
            vals = [d.get(a) for a in cols]
            avg = sum(v for v in vals if v is not None) / max(
                len([v for v in vals if v is not None]), 1)
            row = {"DS": ds_short[ds], "Method": m}
            row.update({a: _pct(d.get(a)) for a in cols})
            row["Avg"] = _pct(avg)
            rows.append(row)
    df = pd.DataFrame(rows)
    return {"df": df,
            "latex": _latex(df, "Adversarial robustness: TPR (\\%) under seven "
                                "attack methods", "tab:adversarial")}


# --- Table IX: adaptive defense component summary -------------------------- #
def table_adaptive(stats: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """stats[dataset] = {cache_hit_rate, borderline_rate, reclassified_rate,
    tpr_gain_pp, convergence, ...}."""
    rows = [
        ("C1: Cache", "Hit rate (%)", "cache_hit_rate", 100, 1),
        ("C2: Confidence", "Borderline (%)", "borderline_rate", 100, 1),
        ("C2: Confidence", "Reclassified (%)", "reclassified_rate", 100, 1),
        ("C3: Threshold", "Convergence (flows)", "convergence", 1, 0),
        ("All", "TPR gain (pp)", "tpr_gain_pp", 1, 1),
    ]
    out = []
    for comp, metric, key, scale, nd in rows:
        row = {"Component": comp, "Metric": metric}
        for ds in DATASETS:
            v = stats.get(ds, {}).get(key)
            row[ds] = "--" if v is None else f"{scale * v:.{nd}f}"
        out.append(row)
    df = pd.DataFrame(out)
    return {"df": df,
            "latex": _latex(df, "Adaptive defense component evaluation summary",
                            "tab:adaptive")}


def save_all(tables: Dict[str, Dict[str, object]], out_dir: str) -> None:
    """Write every generated table to <out_dir>/<name>.tex and print to console."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for name, t in tables.items():
        with open(os.path.join(out_dir, f"{name}.tex"), "w") as fh:
            fh.write(t["latex"])
        print(f"\n===== {name} =====")
        df = t["df"]
        if isinstance(df, dict):
            for k, v in df.items():
                print(f"-- {k} --"); print(v.to_string(index=False))
        else:
            print(df.to_string(index=False))
