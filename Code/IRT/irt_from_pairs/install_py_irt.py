def debug_pyirt():
    import importlib, inspect

    try:
        import py_irt
    except ImportError:
        print("❌ 没有安装 py_irt")
        return

    print("=== py_irt 基本信息 ===")
    print("文件路径:", getattr(py_irt, "__file__", None))
    print("版本号:", getattr(py_irt, "__version__", "N/A"))
    print("顶层符号（前50个）:", dir(py_irt)[:50])

    # 查找 create_* 函数
    creates = [n for n in dir(py_irt) if "create" in n.lower()]
    if creates:
        print("发现 create_* 函数:", creates)
    else:
        print("❌ 没有发现 create_* 函数")

    # 尝试加载类式 API
    try:
        from py_irt.models import irt_2pl, irt_3pl, irt_4pl
        print("✅ 找到类式 API: irt_2pl / irt_3pl / irt_4pl")
        m = irt_2pl.IrtModel(num_examinees=2, num_items=3)

        print("\n=== irt_2pl.IrtModel 可用方法 ===")
        methods = [n for n in dir(m) if not n.startswith("_")]
        print(methods)

        print("\n候选 add 方法:", [n for n in methods if "add" in n.lower()])
        print("候选训练方法:", [n for n in methods if n in ("fit","train","run")])
        print("候选能力(θ)获取方法:", [n for n in methods if "abil" in n.lower() or "theta" in n.lower()])
        print("候选题目参数获取方法:", [n for n in methods if "item" in n.lower() or "param" in n.lower()])

    except Exception as e:
        print("❌ 导入类式 API 失败:", e)


if __name__ == "__main__":
    import pkgutil

    import py_irt

    print("py_irt file:", getattr(py_irt, "__file__", None))
    print("py_irt submodules:", [m.name for m in pkgutil.iter_modules(py_irt.__path__)])

    # 列出 models 子模块下有哪些实现
    import py_irt.models as M

    print("py_irt.models has:", dir(M))

    # 2PL / 4PL 正确类名与可用方法
    from py_irt.models.two_param_logistic import TwoParamLog
    from py_irt.models.four_param_logistic import FourParamLog

    m2 = TwoParamLog(priors="vague", num_items=3, num_subjects=2, device="cpu")
    m4 = FourParamLog(priors="vague", num_items=3, num_subjects=2, device="cpu")


    def brief(x):
        return [n for n in dir(x) if not n.startswith("_")]


    print("\nTwoParamLog methods:", brief(m2))
    print("  has fit_MCMC?:", hasattr(m2, "fit_MCMC"))
    print("  has predict?:", hasattr(m2, "predict"))

    print("\nFourParamLog methods:", brief(m4))
    print("  has fit_MCMC?:", hasattr(m4, "fit_MCMC"))
    print("  has predict?:", hasattr(m4, "predict"))

    debug_pyirt()
