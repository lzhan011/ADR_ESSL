import os
import random
from typing import List, Tuple
from pysat.solvers import Minisat22


# ========== 你给的检查函数（稍微注释一下） ==========
def check_model_with_minisat(clauses: List[List[int]], model: List[int], N: int):
    # 如果长度不对或者有 0，直接 False
    if N is not None and len(model) != N:
        return False
    if any(x == 0 for x in model):
        return False

    try:
        with Minisat22(bootstrap_with=clauses) as solver:
            return solver.solve(assumptions=model)
    except Exception:
        return False


# ========== DIMACS CNF 读写工具函数 ==========

def read_dimacs(path: str) -> Tuple[int, int, List[List[int]], List[str]]:
    """
    读取 DIMACS CNF 文件。
    返回: (num_vars, num_clauses_header, clauses, header_lines)
    - num_vars: p cnf 行中声明的变量数
    - num_clauses_header: p cnf 行中声明的子句数（原始值）
    - clauses: [[lit1, lit2, ...], ...] 不包含结尾的 0
    - header_lines: 所有以 'c' 开头的注释行（写回时可保留）
    """
    num_vars = None
    num_clauses = None
    clauses = []
    header_lines = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('c'):
                header_lines.append(line)
                continue
            if line.startswith('p'):
                parts = line.split()
                # p cnf <num_vars> <num_clauses>
                num_vars = int(parts[2])
                num_clauses = int(parts[3])
                continue

            # 普通子句行
            lits = list(map(int, line.split()))
            if lits[-1] != 0:
                raise ValueError(f"Clause line does not end with 0: {line}")
            clause = lits[:-1]
            clauses.append(clause)

    if num_vars is None or num_clauses is None:
        raise ValueError("DIMACS file missing 'p cnf' line")

    return num_vars, num_clauses, clauses, header_lines


def write_dimacs(path: str, num_vars: int, clauses: List[List[int]], header_lines: List[str] = None):
    """
    将 CNF 写回 DIMACS 格式。
    - header_lines: 原有的 'c ...' 注释行（可选）
    子句数自动用 len(clauses) 填写。
    """
    if header_lines is None:
        header_lines = []

    with open(path, 'w') as f:
        for line in header_lines:
            f.write(line.rstrip() + '\n')
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for cl in clauses:
            line = " ".join(str(l) for l in cl) + " 0\n"
            f.write(line)


# ========== 三种改成 UNSAT 的方法 ==========

def make_unsat_add_empty_clause(num_vars: int, clauses: List[List[int]]):
    """
    方法1：加入空子句 []，公式立刻 UNSAT。
    """
    clauses.append([])  # 空子句
    return num_vars, clauses


def make_unsat_add_contradictory_unit(num_vars: int, clauses: List[List[int]], var: int = 1):
    """
    方法2：加入一对互相矛盾的 unit 子句：
        var
        -var
    使得公式 UNSAT。
    var 默认选 1，也可以随机选 [1, num_vars] 内的变量。
    """
    clauses.append([var])
    clauses.append([-var])
    return num_vars, clauses


def make_unsat_forbid_vertex_all_colors(num_vars: int,
                                        clauses: List[List[int]],
                                        vertex_index: int,
                                        colors_per_vertex: int = 3):
    """
    方法3：保持图着色语义的情况下，让某个顶点所有颜色都被禁止。
    假定变量映射为：
        顶点 v (从1开始) 对应的颜色变量为：
            (v-1)*colors_per_vertex + 1 ... v*colors_per_vertex
    """
    start = (vertex_index - 1) * colors_per_vertex + 1
    end = vertex_index * colors_per_vertex
    for var in range(start, end + 1):
        clauses.append([-var])
    return num_vars, clauses


# ========== 统一封装：随机选择一种方法，把 SAT 改成 UNSAT ==========

def make_unsat_random(num_vars: int, clauses: List[List[int]], method: str = None):
    """
    随机（或指定）选择一种方法把公式改成 UNSAT。
    可选方法：
        - 'empty_clause'
        - 'contradictory_unit'
        - 'forbid_vertex'
    如果 method=None，则随机选。
    """
    methods = ['empty_clause', 'contradictory_unit', 'forbid_vertex']
    if method is None:
        method = random.choice(methods)

    # 复制一份，避免原始列表被改坏
    clauses = [cl.copy() for cl in clauses]

    if method == 'empty_clause':
        num_vars, clauses = make_unsat_add_empty_clause(num_vars, clauses)

    elif method == 'contradictory_unit':
        # 随机选 [1, num_vars] 内的一个变量
        v = random.randint(1, num_vars)
        num_vars, clauses = make_unsat_add_contradictory_unit(num_vars, clauses, var=v)

    elif method == 'forbid_vertex':
        colors_per_vertex = 3
        num_vertices = num_vars // colors_per_vertex
        v_index = random.randint(1, num_vertices)  # 1-based
        num_vars, clauses = make_unsat_forbid_vertex_all_colors(
            num_vars, clauses, vertex_index=v_index, colors_per_vertex=colors_per_vertex
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return num_vars, clauses, method


# ========== 遍历一个文件夹：对所有 .cnf 文件生成 _UNSAT 版本 ==========

def process_folder(folder: str):
    """
    遍历 folder 中所有 .cnf 文件：
      - 跳过文件名中包含 '_UNSAT' 的
      - 对每个文件随机选择一种方法改成 UNSAT
      - 写回同一文件夹，文件名加 '_UNSAT'
      - 用 Minisat 检查是否 UNSAT
    """
    file_list = sorted(os.listdir(folder))

    for fname in file_list:
        # 只处理 .cnf，且跳过已经是 _UNSAT 的文件
        if not fname.endswith(".cnf"):
            continue
        if "_UNSAT" in fname:
            continue

        path = os.path.join(folder, fname)
        print(f"Processing: {path}")

        num_vars, num_clauses_header, clauses, header_lines = read_dimacs(path)

        # 随机选择一种方法改为 UNSAT
        num_vars_unsat, unsat_clauses, method = make_unsat_random(num_vars, clauses, method=None)
        print(f"  -> chosen method: {method}")

        # 检查是否 UNSAT
        is_sat = check_model_with_minisat(unsat_clauses, [], 0)
        is_unsat = not is_sat
        print(f"  -> SAT? {is_sat}, UNSAT? {is_unsat}")

        if not is_unsat:
            print("  [WARN] Formula is not UNSAT after modification, please check!")
        else:
            print("  [OK] Confirmed UNSAT.")

        # 写出新的 UNSAT CNF 文件，和原文件在同一目录
        base_name, ext = os.path.splitext(fname)  # ('flat30-1', '.cnf')
        out_name = f"{base_name}_UNSAT{ext}"      # 'flat30-1_UNSAT.cnf'
        out_path = os.path.join(folder, out_name)
        write_dimacs(out_path, num_vars_unsat, unsat_clauses, header_lines)
        print(f"  -> written to: {out_path}\n")


# ========== 主程序：处理 flat30-60 目录下所有实例 ==========

if __name__ == "__main__":
    c_root = r"/scratch/lzhan011/Satisfiability_Solvers/Code/flat_graph_colouring/Flat_Graph_Colouring"
    folder = os.path.join(c_root, "flat50-115")   #flat30-60 flat50-115

    print(f"Processing folder: {folder}")
    process_folder(folder)
