# ========= 家族与样式（可复用） =========
CLAUDE_MODELS        = ["claude-3-5-haiku", "claude-3-7-sonnet", "claude-3-opus", "claude-sonnet-4"]
DEEPSEEK_MODELS      = ["deepseek-chat", "deepseek-reasoner"]
OPENAI_SPECIAL_RED   = ["gpt-5", "o1", "o3-mini"]   # 红色实线
OPENAI_OTHER_YELLOW  = ["gpt-4.1", "gpt-4o-latest", "gpt-3.5-turbo-0125"]  # 黄色实线

# 图例/绘制顺序：Claude → DeepSeek → OpenAI(红) → OpenAI(黄)
ORDER_LIST = CLAUDE_MODELS + DEEPSEEK_MODELS + OPENAI_SPECIAL_RED + OPENAI_OTHER_YELLOW

# 家族主色与线型
FAMILY_STYLE = {
    "claude"  : dict(color="#7b2cbf", linestyle="--"),            # 紫色 虚线
    "deepseek": dict(color="#2ca02c", linestyle=(0, (10, 6))),     # 绿色 长虚线
    "openai_red":  dict(color="#d62728", linestyle="-"),           # 红色 实线
    "openai_yel":  dict(color="#ffbf00", linestyle="-"),           # 黄色 实线
}

# 家族内 markers（数量会循环复用）
CLAUDE_TRIANGLES  = ["^", "v", "<", ">"]                           # 不同方向三角形
DEEPSEEK_MARKERS  = ["*", "s"]                                     # 星形、方块
OPENAI_RED_MARKS  = ["o", "D", "X"]                                # 圆、菱形、X
OPENAI_YEL_MARKS  = ["P", "h", "d"]                                # 五边形、六边形、窄菱形

def _build_style_book():
    book = {}
    # Claude: 紫 + 虚线 + 三角
    for i, m in enumerate(CLAUDE_MODELS):
        book[m] = {**FAMILY_STYLE["claude"], "marker": CLAUDE_TRIANGLES[i % len(CLAUDE_TRIANGLES)]}
    # DeepSeek: 绿 + 长虚线 + 星/方
    for i, m in enumerate(DEEPSEEK_MODELS):
        book[m] = {**FAMILY_STYLE["deepseek"], "marker": DEEPSEEK_MARKERS[i % len(DEEPSEEK_MARKERS)]}
    # OpenAI 红组：GPT-5 / o1 / o3-mini
    for i, m in enumerate(OPENAI_SPECIAL_RED):
        book[m] = {**FAMILY_STYLE["openai_red"], "marker": OPENAI_RED_MARKS[i % len(OPENAI_RED_MARKS)]}
    # OpenAI 黄组：其他 OpenAI
    for i, m in enumerate(OPENAI_OTHER_YELLOW):
        book[m] = {**FAMILY_STYLE["openai_yel"], "marker": OPENAI_YEL_MARKS[i % len(OPENAI_YEL_MARKS)]}
    return book

STYLE_BOOK = _build_style_book()

def _which_openai_palette(name_lower: str) -> str:
    # gpt-5 / o1 / o3-mini → 红；其余 → 黄
    if any(k in name_lower for k in ["gpt-5", "o1", "o3-mini"]):
        return "openai_red"
    return "openai_yel"

def get_style(model_name: str, fallback_idx: int = 0):
    """
    返回 {color, linestyle, marker}
    未在 STYLE_BOOK 中的模型，按名称归类并给一个可复用的 fallback marker。
    """
    if model_name in STYLE_BOOK:
        return STYLE_BOOK[model_name]

    name = (model_name or "").lower()
    if "claude" in name:
        fam = "claude"; markers = CLAUDE_TRIANGLES
    elif "deepseek" in name:
        fam = "deepseek"; markers = DEEPSEEK_MARKERS
    else:
        fam = _which_openai_palette(name)  # openai_red / openai_yel
        markers = OPENAI_RED_MARKS if fam == "openai_red" else OPENAI_YEL_MARKS

    return {**FAMILY_STYLE[fam], "marker": markers[fallback_idx % len(markers)]}

def order_models(models: list) -> list:
    """按指定 ORDER_LIST 排序，未知模型排在最后并按字母序。"""
    idx = {m: i for i, m in enumerate(ORDER_LIST)}
    return sorted(models, key=lambda m: (idx.get(m, 10_000), m))
