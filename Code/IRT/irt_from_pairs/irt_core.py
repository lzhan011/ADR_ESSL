import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
# ----------------------------
# Scoring & data preparation
# ----------------------------

def ensure_required_columns(df: pd.DataFrame, needed) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

def make_pair_uid(df: pd.DataFrame) -> pd.Series:
    """
    Build a stable item_id for each pair.
    Prefer explicit 'pair_base_name' if present, else original||fixed.
    """
    if 'pair_base_name' in df.columns:
        return df['pair_base_name'].astype(str)
    return df['original_file'].astype(str) + '||' + df['fixed_file'].astype(str)

def score_ADR_strict(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Input all_pairs with columns at least:
      ['model','N','original_prediction','fixed_prediction','is_satisfied', 'original_file','fixed_file' (or pair_base_name)]
    Output responses DataFrame with columns:
      ['model','item_id','N','y']
    where y=1 iff (~original_prediction) & fixed_prediction & is_satisfied
    """
    need = ['model','N','original_prediction','fixed_prediction','is_satisfied']
    ensure_required_columns(df_pairs, need)
    df = df_pairs.copy()

    # normalize booleans
    def _to_bool(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.strip().str.lower().isin(['true','1','yes','y','t'])
    for col in ['original_prediction','fixed_prediction','is_satisfied']:
        df[col] = _to_bool(df[col])

    df['item_id'] = make_pair_uid(df)

    # ADR-Strict
    # df['y'] = (~df['original_prediction']) & (df['fixed_prediction']) & (df['is_satisfied'])
    df['y'] = (~df['original_prediction']) & (df['fixed_prediction'])
    out = df[['model','item_id','N','y']].copy()
    out['y'] = out['y'].astype(int)
    return out


import pandas as pd

def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true","1","yes","y","t"])

def score_instance_level(
    df_pairs: pd.DataFrame,
    drop_if_not_satisfied: bool = True
) -> pd.DataFrame:
    """
    Convert pair rows into instance-level responses for IRT.

    Input df_pairs must contain at least:
      ['model','original_file','original_prediction',
       'fixed_file','fixed_prediction','is_satisfied']

    Output columns:
      ['model','item_id','y']

    Semantics:
      - Keep only rows where is_satisfied==True (default).
      - Truth:  original instance -> False (UNSAT), fixed instance -> True (SAT).
      - y = 1 iff prediction == truth for that instance.
    """
    need = ["model","original_file","original_prediction",
            "fixed_file","fixed_prediction","is_satisfied"]
    missing = [c for c in need if c not in df_pairs.columns]
    if missing:
        raise KeyError(f"[instance scoring] Missing columns: {missing}. "
                       f"Available: {list(df_pairs.columns)}")

    df = df_pairs.copy()

    # normalize booleans
    df["original_prediction"] = _to_bool_series(df["original_prediction"])
    df["fixed_prediction"]    = _to_bool_series(df["fixed_prediction"])
    df["is_satisfied"]        = _to_bool_series(df["is_satisfied"])

    if drop_if_not_satisfied:
        df = df[df["is_satisfied"]].copy()

    if df.empty:
        # Return an empty, well-typed frame
        return pd.DataFrame(columns=["model","item_id","y"])

    # Build two instance rows per pair: original & fixed
    orig = pd.DataFrame({
        "model":   df["model"].astype(str).values,
        "item_id": df["original_file"].astype(str).values,   # each original file is an item
        "pred":    df["original_prediction"].values,
        "truth":   False,   # original should be UNSAT when is_satisfied==True
    })

    fixed = pd.DataFrame({
        "model":   df["model"].astype(str).values,
        "item_id": df["fixed_file"].astype(str).values,      # each fixed file is an item
        "pred":    df["fixed_prediction"].values,
        "truth":   True,    # fixed should be SAT when is_satisfied==True
    })

    inst = pd.concat([orig, fixed], ignore_index=True)

    # If you kept any rows where is_satisfied==False, you could set truth to NaN and drop them;
    # here we already filtered, so compute correctness directly:
    inst["y"] = (inst["pred"] == inst["truth"]).astype(int)

    out = inst[["model","item_id","y"]].copy()
    return out


def responses_to_matrix(df_resp: pd.DataFrame) -> Tuple[list, list, list]:
    """
    Convert responses to sparse triples for py-irt:
    returns (examinees, items, triples[(e_idx, q_idx, y)])
    examinee_id = model (no decode in your table; extend here if you add one)
    """
    ensure_required_columns(df_resp, ['model','item_id','y'])
    df = df_resp.copy()
    examinees = df['model'].astype(str).unique().tolist()
    items = df['item_id'].astype(str).unique().tolist()
    e2i = {e:i for i,e in enumerate(examinees)}
    q2i = {q:i for i,q in enumerate(items)}
    triples = [(e2i[r['model']], q2i[r['item_id']], int(r['y'])) for _, r in df.iterrows()]
    return examinees, items, triples

# ----------------------------
# IRT fitting via py-irt (if present)
# ----------------------------

def try_import_pyirt():
    try:
        import py_irt as pyirt
        return pyirt
    except Exception:
        return None

def fit_pyirt(df_resp: pd.DataFrame, model_type: str = "2PL", seed: int = 0, pair_or_instance_level='pair') -> Dict[str, pd.DataFrame]:
    """
    统一用 py-irt CLI 训练 2PL/3PL/4PL：
      1) 把 df_resp(长表: model,item_id,y) 聚合成 jsonlines（每 subject 一行，带 "responses"）
      2) 调用: python -m py_irt.cli train {2pl|3pl|4pl} <jsonl> <work_dir>
      3) 解析 <work_dir>/best_parameters.json → abilities 与 item_params
    """
    import os, json, tempfile, subprocess, sys
    import pandas as pd

    mt = str(model_type).lower()
    if mt not in {"2pl", "3pl", "4pl"}:
        raise ValueError("model_type must be one of {'2PL','3PL','4PL'}")

    need = {"model", "item_id", "y"}
    if not need.issubset(df_resp.columns):
        raise KeyError(f"fit_pyirt: 缺少必要列 {need - set(df_resp.columns)}，当前列：{list(df_resp.columns)}")

    # 规范 y 为 0/1
    df = df_resp[["model", "item_id", "y"]].copy()
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).clip(0, 1)

    subjects = df["model"].astype(str).nunique()
    items    = df["item_id"].astype(str).nunique()
    if subjects == 0 or items == 0:
        return {}

    def _parse_best_params(best_json: str):
        with open(best_json, "r", encoding="utf-8") as f:
            P = json.load(f)
        abil = pd.DataFrame([{"model": k, "theta": v} for k, v in P.get("subject_ability", {}).items()])
        rows = []
        for q, d in P.get("item_params", {}).items():
            row = {"item_id": q}
            for k in ("a","alpha","discrimination"):
                if k in d: row["a"] = d[k]
            for k in ("b","beta","difficulty"):
                if k in d: row["b"] = d[k]
            for k in ("c","g","guessing","lower_asymptote"):
                if k in d: row["c"] = d[k]
            for k in ("d","u","upper_asymptote"):
                if k in d: row["d"] = d[k]
            rows.append(row)
        items_df = pd.DataFrame(rows)
        return abil, items_df

    try:
        with tempfile.TemporaryDirectory() as tmpd:
            # 写入聚合 JSONL
            jsonl = os.path.join(tmpd, f"pairs_{mt}.jsonlines")
            # 用已有的聚合函数：先把 df_resp 存成临时 CSV，再复用转换逻辑
            tmp_csv = os.path.join(tmpd, "responses_tmp.csv")
            df.to_csv(tmp_csv, index=False)
            _to_jsonlines_from_responses(tmp_csv, jsonl)

            work  = os.path.join(tmpd, f"pyirt_{mt}")
            os.makedirs(work, exist_ok=True)

            # 调用当前解释器的 py_irt.cli
            cmd = [sys.executable, "-m", "py_irt.cli", "train", mt, jsonl, work]
            epochs = 8000
            if epochs is not None:
                cmd += ["--epochs", str(int(epochs))]

            # 可按版本支持追加训练参数，例如：
            # cmd += ["--epochs", "2000", "--seed", str(int(seed))]
            print("[py-irt CLI]", " ".join(cmd))
            subprocess.run(cmd, check=True)

            # 解析最优参数
            best = os.path.join(work, "best_parameters.json")
            if not os.path.exists(best):
                cand = [os.path.join(work, f) for f in os.listdir(work) if f.endswith(".json")]
                best = cand[0] if cand else None
            if not best:
                return {}

            abilities_df, items_df = _parse_best_params(best)

            if pair_or_instance_level == 'instance':
                # 🔹 在这里加方向对齐
                if abilities_df["theta"].mean() < 0:
                    abilities_df["theta"] *= -1
                    if "b" in items_df.columns:
                        items_df["b"] *= -1
            return {
                "abilities": abilities_df,
                "items": items_df,
                "meta": {"n_models": subjects, "n_items": items, "work_dir": work}
            }

    except subprocess.CalledProcessError as e:
        print(f"[py-irt CLI] 训练失败：{e}")
        return {}
    except Exception as e:
        print(f"[fit_pyirt] 异常：{e}")
        return {}





# ----------------------------
# Diagnostics & derived metrics
# ----------------------------

def _get_param_cols(item_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    # Try to infer column names for a,b,c,d across possible naming schemes
    name_map = {"a": None, "b": None, "c": None, "d": None}
    # Common variants
    candidates = {
        "a": ["a", "discrimination", "alpha"],
        "b": ["b", "difficulty", "beta"],
        "c": ["c", "guessing", "lower_asymptote"],
        "d": ["d", "upper_asymptote"]
    }
    for k, opts in candidates.items():
        for c in opts:
            if c in item_df.columns:
                name_map[k] = c
                break
    return name_map

def predict_prob(df_resp: pd.DataFrame, abilities: pd.DataFrame, items: pd.DataFrame, model_type: str) -> pd.Series:
    """
    Compute predicted P(correct) for each row in df_resp using fitted params.
    """
    name_map = _get_param_cols(items)
    a_col, b_col, c_col, d_col = name_map["a"], name_map["b"], name_map["c"], name_map["d"]
    if a_col is None or b_col is None:
        raise ValueError("Cannot find a/b parameter columns in item table. Found columns: %s" % list(items.columns))

    item_lookup = items.set_index("item_id")
    abil_lookup = abilities.set_index("model")

    def _row_prob(r):
        theta = abil_lookup.loc[r["model"], "theta"]
        pars = item_lookup.loc[r["item_id"]]
        a = float(pars[a_col])
        b = float(pars[b_col])
        # base 2PL
        p = 1.0/(1.0 + math.exp(-a*(theta - b)))
        if model_type.upper() in ("3PL","4PL") and c_col in item_lookup.columns:
            c = float(pars.get(c_col, 0.0))
            p = c + (1.0 - c)*p
        if model_type.upper() == "4PL" and d_col in item_lookup.columns:
            d = float(pars.get(d_col, 1.0))
            # Blend lower and upper asymptotes
            c = float(pars.get(c_col, 0.0)) if c_col in item_lookup.columns else 0.0
            p = c + (d - c)* (1.0/(1.0 + math.exp(-a*(theta - b))))
        return p

    return df_resp.apply(_row_prob, axis=1)

def auc_from_probs(y_true: pd.Series, p: pd.Series) -> float:
    # Simple AUC implementation (no sklearn dependency)
    # Rank-based AUC
    df = pd.DataFrame({"y": y_true.astype(int), "p": p})
    df = df.sort_values("p")
    n0 = (df["y"] == 0).sum()
    n1 = (df["y"] == 1).sum()
    if n0 == 0 or n1 == 0:
        return float("nan")
    # Compute rank sum of positives
    df["rank"] = np.arange(1, len(df)+1)
    rank_sum_pos = df.loc[df["y"] == 1, "rank"].sum()
    auc = (rank_sum_pos - n1*(n1+1)/2) / (n0*n1)
    return float(auc)

def item_information_2pl(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    # IIF for 2PL
    P = 1.0/(1.0 + np.exp(-a*(theta - b)))
    Q = 1.0 - P
    return (a**2) * P * Q

def test_information_2pl(theta: np.ndarray, items: pd.DataFrame) -> np.ndarray:
    # Sum of item informations (2PL approx; ignores c/d)
    name_map = _get_param_cols(items)
    a_col, b_col = name_map["a"], name_map["b"]
    info = np.zeros_like(theta, dtype=float)
    for _, row in items.iterrows():
        a = float(row[a_col]); b = float(row[b_col])
        info += item_information_2pl(theta, a, b)
    return info

# ----------------------------
# Top-level runner
# ----------------------------

import os, json, pandas as pd, numpy as np

def _to_jsonlines_from_responses(responses_csv: str, out_jsonl: str) -> str:
    """
    把 responses_*.csv (列: model,item_id,y) 转成 py-irt CLI 需要的 jsonlines：
      每个 subject(=model) 一行：
      {"subject_id": <model>, "responses": {<item_id>: <0/1>, ...}}
    """
    import json, pandas as pd

    df = pd.read_csv(responses_csv, dtype={"model": str, "item_id": str})
    need = {"model", "item_id", "y"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[jsonl] {responses_csv} 缺列: {miss}")

    # 规范 y 为 0/1 int
    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df = df.assign(y=y)

    # 按 subject 分组，聚合为 {item_id: y}
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for subject, g in df.groupby("model", sort=False):
            # 同一 subject 对同一 item 多次，取最后一次；如需“任一正确就算 1”可改成 groupby(...).y.max()
            resp_map = {row.item_id: int(row.y) for row in g.itertuples(index=False)}
            w.write(json.dumps({"subject_id": str(subject), "responses": resp_map}, ensure_ascii=False) + "\n")

    return out_jsonl



def fit_pyirt_via_cli(responses_csv: str, model_type: str, out_dir: str):
    """
    用 py-irt CLI 跑 3PL/4PL（以及需要时的 2PL），输出 best_parameters.json，
    再抽成 abilities_* 与 item_params_* 两个 CSV。
    """
    os.makedirs(out_dir, exist_ok=True)
    model_type = model_type.lower()  # '2pl' | '3pl' | '4pl'
    jsonl = os.path.join(out_dir, f"pairs_{model_type}.jsonlines")
    _to_jsonlines_from_responses(responses_csv, jsonl)

    # 1) 调 CLI 训练
    import subprocess, shlex
    work_dir = os.path.join(out_dir, f"pyirt_{model_type}")
    os.makedirs(work_dir, exist_ok=True)
    cmd = f"py-irt train {model_type} {jsonl} {work_dir}"
    subprocess.run(shlex.split(cmd), check=True)

    # 2) 解析 best_parameters.json
    best = os.path.join(work_dir, "best_parameters.json")
    with open(best, "r", encoding="utf-8") as f:
        params = json.load(f)

    # subjects → θ
    abil_rows = [{"model": k, "theta": v} for k, v in params.get("subject_ability", {}).items()]
    abil_df = pd.DataFrame(abil_rows)
    abil_csv = os.path.join(out_dir, f"abilities_{model_type}.csv")
    abil_df.to_csv(abil_csv, index=False)

    # items → a/b/（+ c/d 如果存在）
    item_rows = []
    for item_id, d in params.get("item_params", {}).items():
        row = {"item_id": item_id}
        # 兼容不同命名：a/alpha、b/beta、c/g、d/u…
        for k in ("a","alpha","discrimination"):
            if k in d: row["a"] = d[k]
        for k in ("b","beta","difficulty"):
            if k in d: row["b"] = d[k]
        for k in ("c","g","guessing","lower_asymptote"):
            if k in d: row["c"] = d[k]
        for k in ("d","u","upper_asymptote"):
            if k in d: row["d"] = d[k]
        item_rows.append(row)
    items_df = pd.DataFrame(item_rows)
    items_csv = os.path.join(out_dir, f"item_params_{model_type}.csv")
    items_df.to_csv(items_csv, index=False)

    return {"abilities": abil_csv, "items": items_csv, "json": best}

def fit_pyirt_two_paramlog(df_resp: pd.DataFrame, num_epochs: int = 1500):
    """
    直接用类 TwoParamLog（2PL）训练（MCMC），不经 CLI。
    返回 abilities DataFrame；item 参数建议仍用 CLI 导出最权威（或改走 R mirt）。
    """
    from py_irt.models.two_param_logistic import TwoParamLog

    df = df_resp[["model","item_id","y"]].copy()
    subj_codes = {s:i for i,s in enumerate(sorted(df["model"].unique()))}
    item_codes = {q:i for i,q in enumerate(sorted(df["item_id"].unique()))}
    df["sid"] = df["model"].map(subj_codes).astype(np.int64)
    df["iid"] = df["item_id"].map(item_codes).astype(np.int64)
    df["y"]   = df["y"].astype(np.int64)

    m = TwoParamLog(priors="vague", num_items=len(item_codes), num_subjects=len(subj_codes), device="cpu")
    m.fit_MCMC(df["sid"].to_numpy(), df["iid"].to_numpy(), df["y"].to_numpy(), num_epochs=num_epochs)

    # 用预测的 logit 平均作为 θ 的近似（官方未暴露 θ 向量接口）
    # 你也可以 m.export(...) 再解析 JSON 取更标准的能力估计
    ability_rows = []
    eps = 1e-6
    for s_name, s_idx in subj_codes.items():
        sub = df[df["sid"] == s_idx]
        p = m.predict(sub["sid"].to_numpy(), sub["iid"].to_numpy())
        p = np.clip(p, eps, 1 - eps)
        theta_hat = float(np.mean(np.log(p/(1-p))))
        ability_rows.append({"model": s_name, "theta": theta_hat})
    abilities = pd.DataFrame(ability_rows)
    return abilities

# ===== CLI 一键拟合器（添加到 irt_core.py）===================================
import os, json, subprocess, shlex
import pandas as pd
import numpy as np


def extract_best_params_from_current_pyirt(best_json_path: str, pair_or_instance_level):
    """
    兼容当前 py-irt: 顶层键可能是 list/tuple 或 dict。
    - abilities: ability + subject_ids
    - items: disc(a) + diff(b) + lambdas(c/d) + item_ids
    """
    import json
    import pandas as pd
    import numpy as np

    def _to_list(x):
        """把 x 规整成 list。支持 list/tuple/np.ndarray/dict/标量。"""
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            # 尝试按数值键排序，否则保持插入顺序的 values
            try:
                # 部分 JSON 会把键写成 "0","1","2"…
                keys = sorted(x.keys(), key=lambda k: int(k))
                return [x[k] for k in keys]
            except Exception:
                return list(x.values())
        if x is None:
            return []
        # 标量
        return [x]

    with open(best_json_path, "r", encoding="utf-8") as f:
        P = json.load(f)

    # ---- abilities ----
    abil  = _to_list(P.get("ability", []))
    subj  = _to_list(P.get("subject_ids", []))
    if len(subj) == len(abil) and len(abil) > 0:
        abilities_df = pd.DataFrame({"model": list(map(str, subj)), "theta": pd.to_numeric(abil, errors="coerce")})
    else:
        # 兜底
        abilities_df = pd.DataFrame([{"model": f"subj_{i}", "theta": v} for i, v in enumerate(abil)])

    # ---- items ----
    disc  = _to_list(P.get("disc", []))   # a
    diff  = _to_list(P.get("diff", []))   # b
    iids  = _to_list(P.get("item_ids", []))

    lambdas = P.get("lambdas", None)
    lamL = _to_list(lambdas) if lambdas is not None else None

    L = max(len(iids), len(disc), len(diff), (len(lamL) if lamL is not None else 0))
    rows = []
    for i in range(L):
        item_id = iids[i] if i < len(iids) else f"item_{i}"
        row = {"item_id": str(item_id)}
        if i < len(disc): row["a"] = disc[i]
        if i < len(diff): row["b"] = diff[i]

        if lamL is not None and i < len(lamL):
            li = lamL[i]
            # li 可能是标量(c) 或 序列([c,d])
            if isinstance(li, (list, tuple)):
                if len(li) >= 1: row["c"] = li[0]
                if len(li) >= 2: row["d"] = li[1]
            else:
                row["c"] = li

        rows.append(row)

    items_df = pd.DataFrame(rows)
    # 类型清洗
    if not items_df.empty:
        for k in ("a","b","c","d"):
            if k in items_df.columns:
                items_df[k] = pd.to_numeric(items_df[k], errors="coerce")
        items_df["item_id"] = items_df["item_id"].astype(str)
    if pair_or_instance_level == 'instance':
        if not abilities_df.empty and abilities_df["theta"].mean() < 0:
            abilities_df["theta"] *= -1
            if "b" in items_df.columns:
                items_df["b"] *= -1
    return abilities_df, items_df


def _fit_one_via_cli(responses_csv: str, model_tag: str, out_dir: str) -> dict:
    import os, json, subprocess, sys
    import pandas as pd

    model_tag = model_tag.lower()
    os.makedirs(out_dir, exist_ok=True)

    jsonl = os.path.join(out_dir, f"pairs_{model_tag}.jsonlines")
    _to_jsonlines_from_responses(responses_csv, jsonl)

    work_dir = os.path.join(out_dir, f"pyirt_{model_tag}")
    os.makedirs(work_dir, exist_ok=True)

    # 用当前解释器，避免 PATH 找不到 py-irt
    cmd = [sys.executable, "-m", "py_irt.cli", "train", model_tag, jsonl, work_dir]
    print("[py-irt CLI]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    best = os.path.join(work_dir, "best_parameters.json")
    if not os.path.exists(best):
        # 兜底找第一个 .json
        cand = [os.path.join(work_dir, f) for f in os.listdir(work_dir) if f.endswith(".json")]
        best = cand[0] if cand else None
    if not best:
        return {}

    # 解析 best_parameters.json → abilities / items
    with open(best, "r", encoding="utf-8") as f:
        P = json.load(f)

    abil = pd.DataFrame([{"model": k, "theta": v} for k, v in P.get("subject_ability", {}).items()])
    rows = []
    for q, d in P.get("item_params", {}).items():
        row = {"item_id": q}
        for k in ("a","alpha","discrimination"):
            if k in d: row["a"] = d[k]
        for k in ("b","beta","difficulty"):
            if k in d: row["b"] = d[k]
        for k in ("c","g","guessing","lower_asymptote"):
            if k in d: row["c"] = d[k]
        for k in ("d","u","upper_asymptote"):
            if k in d: row["d"] = d[k]
        rows.append(row)
    items_df = pd.DataFrame(rows)

    abil_csv  = os.path.join(out_dir, f"abilities_{model_tag}.csv")
    items_csv = os.path.join(out_dir, f"item_params_{model_tag}.csv")
    abil.to_csv(abil_csv, index=False)
    items_df.to_csv(items_csv, index=False)

    return {"abilities": abil_csv, "items": items_csv, "json": best, "work_dir": work_dir}

def fit_all_via_cli(out_dir: str, which = ("2pl","3pl","4pl")) -> dict:
    """
    一键：按 out_dir 下已有的 responses 文件触发 CLI 拟合。
    - 2PL: 优先用 responses_for_2pl.csv；若没有，则用 responses_adr_strict.csv。
    - 3PL/4PL: 需要 responses_for_3pl.csv / responses_for_4pl.csv；缺则跳过。
    返回 { "2pl": {...}, "3pl": {...}, "4pl": {...} }（存在哪个返回哪个）。
    """
    which = {w.lower() for w in which}
    artifacts = {}
    # 2PL
    if "2pl" in which:
        cand = os.path.join(out_dir, "responses_for_2pl.csv")
        if not os.path.exists(cand):
            alt = os.path.join(out_dir, "responses_adr_strict.csv")
            if os.path.exists(alt):
                cand = alt
        if os.path.exists(cand):
            try:
                artifacts["2pl"] = _fit_one_via_cli(cand, "2pl", out_dir)
            except subprocess.CalledProcessError as e:
                print(f"[warn] 2PL CLI 失败：{e}")
    # 3PL
    if "3pl" in which:
        cand = os.path.join(out_dir, "responses_for_3pl.csv")
        if os.path.exists(cand):
            try:
                artifacts["3pl"] = _fit_one_via_cli(cand, "3pl", out_dir)
            except subprocess.CalledProcessError as e:
                print(f"[warn] 3PL CLI 失败：{e}（若不支持 3PL，可改用 R mirt）")
        else:
            print("[skip] 没找到 responses_for_3pl.csv，跳过 3PL。")
    # 4PL
    if "4pl" in which:
        cand = os.path.join(out_dir, "responses_for_4pl.csv")
        if os.path.exists(cand):
            try:
                artifacts["4pl"] = _fit_one_via_cli(cand, "4pl", out_dir)
            except subprocess.CalledProcessError as e:
                print(f"[warn] 4PL CLI 失败：{e}")
        else:
            print("[skip] 没找到 responses_for_4pl.csv，跳过 4PL。")

    return artifacts
# ========================================================================


def run_irt_from_all_pairs(
    all_pairs_path: str,
    out_dir: str,
    seed: int = 0,
    fit_models = ("2PL","3PL","4PL"),
    pair_or_instance_level = 'pair'
) -> Dict[str, str]:
    """
    End-to-end:
      1) load all_pairs Excel
      2) score ADR-Strict
      3) per model-type: fit -> export abilities/items -> AUC -> TIC plot
    Returns dict of artifact paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    df_pairs = pd.read_excel(all_pairs_path)
    if pair_or_instance_level == 'pair':
        resp = score_ADR_strict(df_pairs)
    elif pair_or_instance_level == 'instance':
        resp = score_instance_level(df_pairs, drop_if_not_satisfied=False)
    resp.to_csv(os.path.join(out_dir, "responses_adr_strict.csv"), index=False)

    artifacts = {"responses": os.path.join(out_dir, "responses_adr_strict.csv")}

    # Basic CTT summaries (useful even if no fitter)
    ctt = resp.groupby(["model"], as_index=False)["y"].agg(total_correct="sum", total_items="size")
    ctt["prop_correct"] = ctt["total_correct"]/ctt["total_items"]
    ctt.to_csv(os.path.join(out_dir, "ctt_summary.csv"), index=False)
    artifacts["ctt"] = os.path.join(out_dir, "ctt_summary.csv")

    # Try fits
    for mtype in fit_models:
        res = fit_pyirt(resp, model_type=mtype, seed=seed, pair_or_instance_level=pair_or_instance_level)
        if not res:
            # Export matrix to help external fitting (e.g., R mirt)
            resp.to_csv(os.path.join(out_dir, f"responses_for_{mtype.lower()}.csv"), index=False)
            artifacts[mtype+"_responses"] = os.path.join(out_dir, f"responses_for_{mtype.lower()}.csv")
            continue

        resp.to_csv(os.path.join(out_dir, f"responses_for_{mtype.lower()}.csv"), index=False)
        artifacts[mtype + "_responses"] = os.path.join(out_dir, f"responses_for_{mtype.lower()}.csv")

        abil = res["abilities"]; items = res["items"]

        abil_out = os.path.join(out_dir, f"abilities_{mtype.lower()}.csv")
        items_out = os.path.join(out_dir, f"item_params_{mtype.lower()}.csv")
        abil.to_csv(abil_out, index=False); items.to_csv(items_out, index=False)
        artifacts[mtype+"_abilities"] = abil_out
        artifacts[mtype+"_items"] = items_out

        # Predictive AUC
        try:
            p = predict_prob(resp, abil, items, mtype)
            auc = auc_from_probs(resp["y"], p)
        except Exception as e:
            auc = float("nan")
        with open(os.path.join(out_dir, f"fit_report_{mtype.lower()}.txt"), "w", encoding="utf-8") as f:
            f.write(f"Models: {res['meta']['n_models']}, Items: {res['meta']['n_items']}\\n")
            f.write(f"AUC (predictive): {auc:.4f}\\n")
        artifacts[mtype+"_report"] = os.path.join(out_dir, f"fit_report_{mtype.lower()}.txt")

        # TIC (2PL approx for display)
        try:
            import matplotlib.pyplot as plt
            thetas = np.linspace(-4, 4, 201)
            tic = test_information_2pl(thetas, items)
            plt.figure()
            plt.plot(thetas, tic)
            plt.xlabel("theta"); plt.ylabel("Test Information")
            plt.title(f"TIC (2PL approx) - {mtype}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"tic_{mtype.lower()}.png"))
            plt.close()
            artifacts[mtype+"_tic"] = os.path.join(out_dir, f"tic_{mtype.lower()}.png")
        except Exception:
            pass

    return artifacts