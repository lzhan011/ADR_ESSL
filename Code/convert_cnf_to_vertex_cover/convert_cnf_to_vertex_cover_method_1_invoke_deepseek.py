from itertools import combinations, chain
import os
import networkx as nx
import re

from openai import base_url


def literal_var(lit: str) -> str:
    """返回文字所属变量名：'x1' 或 '!x1' -> 'x1'。"""
    return lit[1:] if lit.startswith('!') else lit

def literal_is_neg(lit: str) -> bool:
    return lit.startswith('!')

def var_node(var: str, is_pos: bool) -> str:
    """变量器件中的节点名：正文字 v(x) / 负文字 v(!x)。"""
    return f"v({var})" if is_pos else f"v(!{var})"

def clause_node(j: int, t: int) -> str:
    """子句器件中的节点名：c_{j,t}，j 从 0 开始，t ∈ {0,1,2}。"""
    return f"c({j},{t})"

def add_edge(G: dict, u: str, v: str):
    """在无向图邻接表中加入边 (u,v)。"""
    if u == v:
        return
    G.setdefault(u, set()).add(v)
    G.setdefault(v, set()).add(u)

def reduce_3cnf_to_vertex_cover(clauses):
    vars_set = {literal_var(lit) for clause in clauses for lit in clause}
    n = len(vars_set)
    m = len(clauses)
    G = nx.Graph()

    var_nodes = {}
    for x in sorted(vars_set):
        vx_pos = var_node(x, True)
        vx_neg = var_node(x, False)
        var_nodes[x] = (vx_pos, vx_neg)
        G.add_edge(vx_pos, vx_neg)

    clause_nodes = []
    for j, clause in enumerate(clauses):
        assert len(clause) == 3
        cjs = [clause_node(j, t) for t in range(3)]
        clause_nodes.append(cjs)
        G.add_edges_from([(cjs[0], cjs[1]), (cjs[1], cjs[2]), (cjs[0], cjs[2])])
        for t, lit in enumerate(clause):
            x = literal_var(lit)
            node_lit = var_node(x, is_pos=not literal_is_neg(lit))
            G.add_edge(cjs[t], node_lit)

    k = n + 2 * m
    info = {
        "n_vars": n,
        "m_clauses": m,
        "var_nodes": var_nodes,
        "clause_nodes": clause_nodes
    }
    return G, k, info

# ---------- 下面是小型工具：验证某集合是否为顶点覆盖 / 穷举找 <=k 的覆盖 ----------

def is_vertex_cover(G, cover: set) -> bool:
    """检查 cover 是否覆盖 G 的所有边（G 为 networkx.Graph）。"""
    for u, v in G.edges():
        if not (u in cover or v in cover):
            return False
    return True

def vertices(G):
    """返回顶点列表（G 为 networkx.Graph）。"""
    return list(G.nodes())

def find_vertex_cover_upto_k(G, k: int):
    """
    穷举找一个大小 ≤ k 的顶点覆盖（小图测试用）。
    返回: cover(set) 或 None
    """
    V = vertices(G)
    for r in range(k + 1):
        for comb in combinations(V, r):
            S = set(comb)
            if is_vertex_cover(G, S):
                return S
    return None


def exists_vertex_cover_upto_k(G, k: int):
    """决策版本：是否存在大小 ≤ k 的顶点覆盖；返回 (bool, cover_or_None)。"""
    cover = find_vertex_cover_upto_k(G, k)
    return (cover is not None), cover


def graph_as_sets(G):
    """把 NetworkX 图转成 (V, E) 集合形式。"""
    V = set(G.nodes())
    E = {tuple(sorted(e)) for e in G.edges()}
    return V, E


# ---- 快速决策：Vertex Cover ≤ k ----
# 需要: networkx; （可选）python-sat
from networkx.algorithms.matching import max_weight_matching

def vc_decide_fast(G, k, use_sat=True):
    """
    决策是否存在大小 ≤ k 的顶点覆盖。
    优先: 匹配下界 / 2-近似 剪枝；若未定，再走 SAT（或回溯）。
    返回: (ok: bool, cover: set or None)
    """
    # 0) 无边图
    if G.number_of_edges() == 0:
        return True, set()

    # 1) 最大匹配下界剪枝 & 2-近似上界
    # maximum cardinality matching on general graph
    M = max_weight_matching(G, maxcardinality=True)
    lb = len(M)  # lower bound
    if lb > k:
        return False, None

    # a simple 2-approx cover from matched edges
    approx_cover = set()
    for u, v in M:
        approx_cover.add(u); approx_cover.add(v)
    if len(approx_cover) <= k:
        return True, approx_cover

    # 2) SAT 路径（强烈推荐）
    if use_sat:
        try:
            from pysat.formula import CNF
            from pysat.card import CardEnc
            from pysat.solvers import Minisat22
        except Exception as e:
            # 没装 python-sat，则退化到分支
            return _vc_branch_and_bound(G, k)

        # 变量映射
        nodes = list(G.nodes())
        vid = {u: i+1 for i, u in enumerate(nodes)}  # SAT 变量从 1 开始

        cnf = CNF()
        # 边约束: y_u ∨ y_v
        for u, v in G.edges():
            cnf.append([vid[u], vid[v]])
        # at-most-k: sum y_u ≤ k
        # amk = CardEnc.atmost(lits=[vid[u] for u in nodes], bound=k, encoding='seqcounter')
        amk = CardEnc.atmost(lits=[vid[u] for u in nodes], bound=k, encoding=1)  # 1 对应 seqcounter

        cnf.extend(amk.clauses)

        with Minisat22(bootstrap_with=cnf.clauses) as m:
            sat = m.solve()
            if not sat:
                return False, None
            model = set(l for l in m.get_model() if l > 0)
            cover = {u for u in nodes if vid[u] in model}
            return True, cover

    # 3) 备选：小而精的分支搜索（比穷举快得多）
    return _vc_branch_and_bound(G, k)


def _vc_branch_and_bound(G, k):
    """
    简洁分支+剪枝（选一条未覆盖边 (u,v)，分支包含 u 或包含 v）。
    用最大匹配下界作剪枝。适合小中规模或无 SAT 可用时。
    """
    # 剪枝：k < 0 则失败；无边则成功
    if k < 0:
        return False, None
    if G.number_of_edges() == 0:
        return True, set()

    # 下界剪枝（最大匹配）
    M = max_weight_matching(G, maxcardinality=True)
    if len(M) > k:
        return False, None

    # 取一条边分支
    u, v = next(iter(G.edges()))
    # 分支1: 选 u
    G1 = G.copy()
    G1.remove_node(u)
    ok1, cov1 = _vc_branch_and_bound(G1, k-1)
    if ok1:
        cov1 = set(cov1); cov1.add(u)
        return True, cov1

    # 分支2: 选 v
    G2 = G.copy()
    G2.remove_node(v)
    ok2, cov2 = _vc_branch_and_bound(G2, k-1)
    if ok2:
        cov2 = set(cov2); cov2.add(v)
        return True, cov2

    return False, None



def demo():
    # 例子： (x1 ∨ !x2 ∨ x3) ∧ (!x1 ∨ x2 ∨ !x3)
    clauses = [
        ["x1", "!x2", "x3"],
        ["!x1", "x2", "!x3"],
    ]
    G, k, info = reduce_3cnf_to_vertex_cover(clauses)
    V, E = graph_as_sets(G)
    print(f"#vars={info['n_vars']}, #clauses={info['m_clauses']}, k={k}")
    # 展示部分图信息
    print("Variable gadgets:")
    for x, (vp, vn) in info["var_nodes"].items():
        print(f"  {x}: {vp} -- {vn}")
    print("Clause gadgets (triangles):")
    for j, tri in enumerate(info["clause_nodes"]):
        print(f"  C{j}: {tri}")

    # 小规模验证（仅用于 demo；大实例请勿穷举）
    cover = find_vertex_cover_upto_k(G, k)
    print(f"Found cover of size ≤ k? {cover is not None}")
    if cover is not None:
        print(f"Example cover size = {len(cover)}")


def read_dimacs(filepath):
    clauses = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('c') or line.startswith('p'):
                continue
            parts = line.strip().split()
            # if parts and parts[-1] == '0':
            if parts and parts[-1] == '0' and len(parts) == 4 and re.fullmatch(r"[\d\-]", line[0]):

                clause = list(map(int, parts[:-1]))
                clauses.append(clause)
    return clauses


import re

# --- 如果不是 3-CNF，用这个把任意 CNF 拆成 3-CNF（等可满足性；经典链式法） ---
def to_3cnf_from_int_clauses(clauses):
    """
    clauses: List[List[int]] 任意长度的子句（整数文字，正=变量，负=取反）
    return: clauses3 (List[List[int]])  -> 3-CNF
    """
    if not clauses:
        return []
    max_var = max(abs(l) for C in clauses for l in C) if clauses else 0
    next_var = max_var + 1
    clauses3 = []
    for C in clauses:
        t = len(C)
        if t <= 3:
            clauses3.append(C[:])
        else:
            y = next_var; next_var += 1
            clauses3.append([C[0], C[1], y])             # (l1 ∨ l2 ∨ y1)
            for i in range(2, t - 2):
                y_next = next_var; next_var += 1
                clauses3.append([-y, C[i], y_next])      # (¬y_i ∨ l_{i+1} ∨ y_{i+1})
                y = y_next
            clauses3.append([-y, C[-2], C[-1]])          # (¬y_{t-3} ∨ l_{t-1} ∨ l_t)
    return clauses3

def int_lit_to_str(lit: int) -> str:
    """把整数文字转成 'xN' 或 '!xN' 字符串形式。"""
    return f"x{lit}" if lit > 0 else f"!x{-lit}"

def int_clauses_to_str_3cnf(clauses):
    """
    若已是 3-CNF：直接逐子句长度检查，否则先 to_3cnf，再转字符串。
    返回：List[List[str]]，每个子句长度为3，元素如 'x1' 或 '!x2'
    """
    all_len3 = all(len(C) == 3 for C in clauses)
    if not all_len3:
        clauses = to_3cnf_from_int_clauses(clauses)
    # 现在确保是 3-CNF
    assert all(len(C) == 3 for C in clauses), "to_3cnf 失败：仍存在非3子句"
    return [[int_lit_to_str(l) for l in C] for C in clauses]

# --- 用 DIMACS 路径读取 → 构图 → 求是否有 ≤k 的顶点覆盖 ---
def run_vc_pipeline_from_dimacs(filepath):
    # 1) 读取 DIMACS
    clauses_int = read_dimacs(filepath)     # 形如 [[1,-2,3], ...]
    # 2) 转成 3-CNF 且文字为 'xN'/'!xN'
    clauses_str_3 = int_clauses_to_str_3cnf(clauses_int)
    # 3) 归约成 VC
    G, k, info = reduce_3cnf_to_vertex_cover(clauses_str_3)
    # 4) 小规模验证顶点覆盖（穷举）
    # ok, cover = exists_vertex_cover_upto_k(G, k)
    ok, cover = vc_decide_fast(G, k, use_sat=True)
    return ok, cover, G, k, info

def save_llm_answer(answer_text: str,
                    out_dir: str,
                    base_filename: str,
                    model_selected: str,
                    suffix_template: str = "_llm_answer_{model}.txt") -> str:
    """
    将 LLM 的回答保存到文件：
    - out_dir: 建议传 prompt 输出目录（例如 prompt_out_dir）
    - 实际保存到 out_dir/answers/ 子目录
    - 文件名：<原cnf名去后缀>_llm_answer_<model>.txt
    返回：保存文件的完整路径
    """
    answers_dir = os.path.join(out_dir, "answers")
    os.makedirs(answers_dir, exist_ok=True)
    suffix = suffix_template.format(model=model_selected.replace("/", "_"))
    out_path = os.path.join(answers_dir, os.path.splitext(base_filename)[0] + suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(answer_text if answer_text is not None else "")
    return out_path

import json
import re

def parse_llm_answer_json(ans_text: str):
    """
    解析 LLM 返回的一行 JSON：
    期望格式: {"answer":"YES"|"NO","cover":[...]}
    返回: ok(bool), answer_bool(bool|None), cover_list(list[int])
      - ok=False 表示解析失败
    """
    if not ans_text:
        return False, None, []

    # 有些模型可能意外输出多行，这里抓取第一段 JSON
    # 尝试匹配最外层花括号
    m = re.search(r"\{.*\}", ans_text, flags=re.DOTALL)
    if m:
        s = m.group(0)
    else:
        s = ans_text.strip()

    try:
        obj = json.loads(s)
    except Exception:
        return False, None, []

    answer = obj.get("answer", "").strip().upper()
    cover = obj.get("cover", [])
    if answer not in ("YES", "NO"):
        return False, None, []
    if answer == "YES":
        # 规范化 cover：整数、去重、排序
        try:
            cover = sorted({int(x) for x in cover})
        except Exception:
            return False, None, []
        return True, True, cover
    else:
        return True, False, []


# --- 在你的批处理中调用示例 ---
def process_one_file(filepath, prompt_out_dir=None, model_selected=""):
    print("\n\n*******")
    ok, cover, G, k, info = run_vc_pipeline_from_dimacs(filepath)

    # 结果打印（保持你现有逻辑）
    if ok:
        print(f"[{os.path.basename(filepath)}] SAT? True  (⇔ 存在大小 ≤ {k} 的 VC)")
    else:
        print(f"[{os.path.basename(filepath)}] SAT? False 不存在大小 ≤ {k} 的 VC")
    print(f"k = {k}, |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
    if cover is not None:
        print(f"Example vertex cover size = {len(cover)}")

    # === 新增：导出给 LLM 的 (V,E,k) Prompt 文件 ===
    # 将图重标号为整数并排序
    V, E, mapping = graph_to_sets_sorted(G, relabel_to_int=True)
    prompt_text = make_vc_prompt(V, E, k, instance_name=os.path.basename(filepath))

    # 保存 prompt 文件（缺省放到与 output_dir 同级，你也可自定义）
    if prompt_out_dir is None:
        prompt_out_dir = os.path.join(os.path.dirname(filepath), "vc_prompts")
    out_path = save_prompt_for_instance(prompt_text, prompt_out_dir, os.path.basename(filepath))

    print(f"Prompt saved to: {out_path}")

    # 如果需要随后把这个 prompt 读出交给 LLM，用：
    prompt_for_llm = load_prompt(out_path)
    answer_text = send_to_llm(prompt_for_llm, model_selected)  # 或传 api_key="..."
    if answer_text is not None:
        print("LLM Response:\n", answer_text)
        # === 保存 LLM 回复到文件 ===
        ans_path = save_llm_answer(
            answer_text=answer_text,
            out_dir=prompt_out_dir,  # 就放在 prompt 目录的 answers/ 子目录
            base_filename=os.path.basename(filepath),
            model_selected=model_selected
        )
        print(f"LLM answer saved to: {ans_path}")
    ok_parse, is_yes, cover_llm = parse_llm_answer_json(answer_text)

    cover_llm_valid = False
    if ok_parse and is_yes:
        # 用你的 NetworkX 图 G 和阈值 k 校验
        valid = (len(cover_llm) <= k) and is_vertex_cover(G, set(cover_llm))
        if not valid:
            cover_llm_valid = False
            # 计入错误：这次回答按 NO/错误处理
        else:
            cover_llm_valid = True

    return ok_parse, is_yes, cover_llm, ok, cover_llm_valid


# ---------- 将 networkx 图转 (V,E) 并可选重标号为整数 ----------
def graph_to_sets_sorted(G, relabel_to_int=True):
    """
    将 networkx.Graph 转成 (V, E) 集合表示，并进行稳定排序。
    relabel_to_int=True 时，用 1..|V| 的整数重标号节点，返回映射。
    返回: V_sorted(list), E_sorted(list of tuple), mapping(dict or None)
    """
    if relabel_to_int:
        nodes = sorted(G.nodes(), key=lambda x: str(x))
        mapping = {u: i+1 for i, u in enumerate(nodes)}
        H = nx.relabel_nodes(G, mapping, copy=True)
        V = sorted(H.nodes())
        E = sorted([tuple(sorted(e)) for e in H.edges()])
        return V, E, mapping
    else:
        V = sorted(G.nodes(), key=lambda x: str(x))
        E = sorted([tuple(sorted(e)) for e in G.edges()],
                   key=lambda e: (str(e[0]), str(e[1])))
        return V, E, None

# ---------- 生成 Vertex Cover 决策题的 Prompt 文本 ----------
def make_vc_prompt(V, E, k, instance_name=None):
    """
    V: list（已排序）
    E: list of 2-tuples（已排序，端点已排序）
    k: int
    返回: 字符串 prompt
    """
    V_str = "{ " + ", ".join(str(v) for v in V) + " }"
    E_str = "{ " + ", ".join("{" + f"{u}, {v}" + "}" for (u, v) in E) + " }"
    header = (
        f"You are an expert on graph vertex cover problem. "
        f"The following is an undirected graph represented as a pair where "
        f"the first component is the set of vertices and the second is a set of edges. "
        f"Given this graph, determine whether there exists a vertex cover of size ≤ {k}.\n\n"
        "IMPORTANT OUTPUT FORMAT:\n"
        "Return ONLY a single-line JSON object with keys:\n"
        '  - \"answer\": either \"YES\" or \"NO\"\n'
        '  - \"cover\": a list of distinct integers (the vertex set) if and only if \"answer\" is \"YES\"; otherwise []\n'
        '  - \"explain\": a short plain-text explanation (max 1–2 sentences)\n'
        "Rules:\n"
        f"  * Output \"YES\" ONLY IF you can provide a concrete vertex set C with |C| ≤ {k} that covers EVERY edge in E.\n"
        "  * If you are unsure OR cannot verify ALL edges are covered OR cannot list such C, output \"NO\" and set cover=[].\n"
        "  * No text outside the JSON. One line only.\n"
        "Example valid outputs:\n"
        f'  {{\"answer\":\"YES\",\"cover\":[1,3,7],\"explain\":\"All edges incident to at least one of 1,3,7.\"}}\n'
        f'  {{\"answer\":\"NO\",\"cover\":[],\"explain\":\"Unable to verify a cover within size {k}.\"}}\n'
    )

    name_line = (f"Instance: {instance_name}\n" if instance_name else "")
    body = (
        f"{name_line}"
        f"Graph (V, E):\n"
        f"V = {V_str}\n"
        f"E = {E_str}\n"
        f"k = {k}\n\n"
        "Task: Decide if there exists a vertex cover of size ≤ k.\n"
        "If and only if YES, return a valid cover set and a brief explanation in the JSON as described."
    )
    return header + "\n" + body


# ---------- 写入/读取 Prompt 文件 ----------
def save_prompt_for_instance(prompt_text, out_dir, base_filename, suffix="_vc_prompt.txt"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(base_filename)[0] + suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    return out_path

def load_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def send_to_llm(prompt_text: str,
                model_selected: str = "gpt-4o",
                system_msg: str = "You are an expert on graph vertex cover problem.",
                temperature: float = 0.0) -> str:
    """
    调用 LLM（OpenAI SDK）获取答案。
    - 优先使用传入的 api_key；否则读取环境变量 OPENAI_API_KEY
    - 返回模型的纯文本回复（失败返回 None）
    """
    # 延迟导入，避免未使用时报依赖错误
    try:
        from openai import OpenAI
    except Exception as e:
        print("[send_to_llm] OpenAI SDK 未安装，请先: pip install openai")
        print("Error:", e)
        return None

    # API Key
    api_base = "https://api.openai.com/v1"
    import os
    # key = os.environ.get("OPENAI_API_KEY")
    key = os.environ["DEEPSEEK_API_KEY"]
    api_base = 'https://api.deepseek.com'
    if not key:
        print("[send_to_llm] 未找到 API Key。请设置环境变量 OPENAI_API_KEY 或通过 api_key 参数传入。")
        return None

    try:
        client = OpenAI(api_key=key, base_url=api_base)
        resp = client.chat.completions.create(
            model=model_selected,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_text}
            ],
            temperature=temperature
        )
        # 兼容不同版本 SDK 的字段访问
        # 新版通常是 resp.choices[0].message.content
        content = None
        choice0 = resp.choices[0]
        if hasattr(choice0, "message") and isinstance(choice0.message, dict):
            content = choice0.message.get("content", None)
        elif hasattr(choice0, "message") and hasattr(choice0.message, "content"):
            content = choice0.message.content
        else:
            # 某些旧版可能是 text 字段
            content = getattr(choice0, "text", None)

        if content is None:
            print("[send_to_llm] 未从响应中解析到文本内容。原始响应：", resp)
            return None
        return content

    except Exception as e:
        print("[send_to_llm] 调用 LLM 失败：", e)
        return None


def read_cnf():
    model_list = ['gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-4-turbo', 'chatgpt-4o-latest', 'gpt-4.1', 'gpt-4o', 'o3-mini']
    model_list = ['deepseek-reasoner']
    O1_input_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/fix_cnf/fixed_set_mul_N'
    output_dir_root = '/work/lzhan011/Satisfiability_Solvers/Code/convert_cnf_to_vertex_cover/vertex_cover_graph'

    for model_selected in model_list:
        # for N in [5, 8, 10, 25, 50, 60]:
        for N in [8, 10, 25, 50, 60]:
            global_total = 0
            global_correct = 0
            global_correct_llm_vc = 0
            N = str(N)
            dir_name = f"unsat_cnf_low_alpha_N_{N}_openai_prediction_o1"
            O1_input_dir = os.path.join(O1_input_dir_root, dir_name)
            output_dir = os.path.join(output_dir_root, dir_name)
            output_dir = output_dir + '_openai_prediction_' + str(model_selected)
            os.makedirs(output_dir, exist_ok=True)

            # 专门放 prompt 的目录
            prompt_out_dir = os.path.join(output_dir, "vc_prompts")
            os.makedirs(prompt_out_dir, exist_ok=True)

            for file in sorted(os.listdir(O1_input_dir)):
                filepath = os.path.join(O1_input_dir, file)
                if not os.path.exists(filepath):
                    continue
                ok_parse, is_yes, cover_llm, ok, cover_llm_valid = process_one_file(filepath, prompt_out_dir=prompt_out_dir, model_selected=model_selected)

                if ok_parse:
                    global_total += 1
                    if is_yes == ok:
                        global_correct += 1
                        print(
                            f"[ACC] Correct so far: {global_correct}/{global_total} = {global_correct / global_total:.3f}")
                    else:
                        print(
                            f"[ACC] Wrong. Correct so far: {global_correct}/{global_total} = {global_correct / global_total:.3f}")

                    if cover_llm_valid:
                        global_correct_llm_vc += 1
                        print(
                            f"[ACC] Correct so far: {global_correct_llm_vc}/{global_total} = {global_correct_llm_vc / global_total:.3f}")

            print(
                f"[ACC] Correct final: {global_correct}/{global_total} = {global_correct / global_total:.3f}")
            print(
                f"[ACC] Wrong. Correct final: {global_correct}/{global_total} = {global_correct / global_total:.3f}")
            print(
                f"[ACC] Correct Final: {global_correct_llm_vc}/{global_total} = {global_correct_llm_vc / global_total:.3f}")




            # # cnf_text = cnf_to_prompt(cnf)
            #
            # #         prompt = f"""You are a SAT logic solver.
            # # Please use a step-by-step method to solve the following 3-CNF formula.
            # #
            # # At each step, record:
            # # * Which variable is assigned
            # # * Any propagated implications
            # # * Whether the formula is satisfied or a conflict occurs
            # #
            # # Finally, output:
            # # * Whether the formula is SATISFIABLE or UNSATISFIABLE
            # # * Number of branches (i.e., decision points)
            # # * Number of conflicts (i.e., backtracking steps)
            # #
            # # The formula is:
            # # {cnf_text}
            # # """
            #
            # prompt = f"""You are a SAT logic solver.
            # Please use a step-by-step method to solve the following 3-CNF formula.
            #
            # Finally, output only the following three items, with no extra explanation::
            # * Whether the formula is SATISFIABLE or UNSATISFIABLE
            # * Number of branches (i.e., decision points)
            # * Number of conflicts (i.e., backtracking steps)
            # If the formula is SATISFIABLE, please give me the value for each literals.
            #
            # The formula is:
            # {cnf_text}
            # """
            #
            # # 调用模型 + 记录时间
            # try:
            #     start_time = time.time()
            #     response = safe_call_chatgpt(prompt, model_selected)
            #     elapsed_time = time.time() - start_time
            #     sat, branches, conflicts = parse_response(response)
            # except Exception as e:
            #     response = f"[Error] {str(e)}"
            #     elapsed_time = 0
            #     sat, branches, conflicts = "Call_API_Error", 0, 0
            #
            # # 写入文件
            #
            # N, L, alpha, inst_idx = find_k_n_alpha(file)
            # with open(os.path.join(output_dir, file), "w", encoding="utf-8") as f:
            #     f.write("c Random 3-SAT\n")
            #     f.write(f"c alpha={round(alpha, 2)}, N={N}, L={L}, instance={inst_idx + 1}\n")
            #     f.write(f"p cnf {N} {L}\n")
            #     for clause in cnf:
            #         f.write(" ".join(str(x) for x in clause) + " 0\n")
            #     f.write(f"\nc GPT solve time: {elapsed_time:.2f} seconds\n\n")
            #     f.write(response.strip())
            #     write_file_number += 1
            #     print("write_file_number:", write_file_number)
            #     print(os.path.join(output_dir, file))


# ------------------ DEMO ------------------
if __name__ == "__main__":
    demo()
    read_cnf()

