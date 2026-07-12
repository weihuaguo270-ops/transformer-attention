"""
实验对比工具 — 查看、筛选、对比实验记录

用法:
  python experiments/runs/compare.py
"""
import json, os, sys, shutil
from datetime import datetime

RUNS_DIR = os.path.dirname(os.path.abspath(__file__))
TW = shutil.get_terminal_size().columns - 2  # 终端可用宽度


def load_all():
    """加载所有实验"""
    experiments = []
    for d in sorted(os.listdir(RUNS_DIR)):
        if not d[0].isdigit():
            continue
        cp = os.path.join(RUNS_DIR, d, "config.json")
        rp = os.path.join(RUNS_DIR, d, "results.json")
        if not os.path.exists(cp) or not os.path.exists(rp):
            continue
        with open(cp, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(rp, "r", encoding="utf-8") as f:
            results = json.load(f)
        experiments.append((d, config, results))
    return experiments


def trunc(s, n):
    """截断"""
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "."


def fmt(v, d=" -"):
    """格式化数值"""
    if v is None:
        return d
    if isinstance(v, (int, float)):
        return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}" if abs(v) < 1e4 else f"{v:.1e}"
    return str(v)


def filter_experiments(experiments, tags=None, ids=None, last=None):
    """筛选实验"""
    if tags:
        tags = [t.lower() for t in tags]
        experiments = [e for e in experiments if any(
            t in [x.lower() for x in e[1].get("tags", [])] for t in tags)]
    if ids:
        filtered = []
        for eid, c, r in experiments:
            for i in ids:
                s = i.zfill(3) if i.isdigit() and len(i) < 3 else i
                if eid.startswith(s) or eid == i:
                    filtered.append((eid, c, r))
                    break
        experiments = filtered
    if last:
        experiments = experiments[-last:]
    return experiments


def fuzzy_find(experiments, query):
    """模糊匹配"""
    q = query.strip().lower()
    if not q:
        return []
    matches = []
    for eid, config, _ in experiments:
        desc = config.get("description", config.get("desc", "")).lower()
        tags = " ".join(config.get("tags", [])).lower()
        if eid.lower().startswith(q) or q in desc or q in tags:
            matches.append(eid)
    return matches


# ── 打印函数（自适应宽度） ────────────────────

def print_table(experiments):
    """打印指标汇总表"""
    if not experiments:
        print("没有匹配的实验。")
        return
    W = min(TW, 120)
    col_id, col_val, col_ppl, col_ep, col_bep, col_gap = 10, 8, 8, 5, 5, 6
    remain = W - col_id - col_val - col_ppl - col_ep - col_bep - col_gap - 14
    col_desc = max(16, remain // 3)
    col_cfg = max(24, remain - col_desc)

    sep = "=" * W
    h = (f"{'ID':>{col_id}}  {'描述':>{col_desc}}  {'模型':>{col_cfg}}  "
         f"{'Val':>{col_val}}  {'PPL':>{col_ppl}}  {'轮':>{col_ep}}  "
         f"{'最':>{col_bep}}  {'差':>{col_gap}}")
    print(f"\n{sep}\n{h}\n{sep}")

    for eid, config, results in experiments:
        desc = trunc(config.get("description", ""), col_desc)
        dm = config.get("d_model", "?")
        lr = config.get("lr", "?")
        st = config.get("data_stories", "?")
        tags = ",".join(config.get("tags", []))[:max(4, col_cfg - 14)]
        cfg = trunc(f"d={dm} lr={lr} |{st}s {tags}", col_cfg)
        bv = results.get("best_val_loss")
        pp = results.get("perplexity")
        ea = results.get("epochs_actual")
        be = results.get("best_epoch") or "-"
        tl = results.get("final_train_loss")
        vl = results.get("final_val_loss")
        gp = round(vl - tl, 2) if isinstance(tl, (int, float)) and isinstance(vl, (int, float)) else "-"
        print(f"{eid:>{col_id}}  {desc:>{col_desc}}  {cfg:>{col_cfg}}  "
              f"{fmt(bv):>{col_val}}  {fmt(pp):>{col_ppl}}  "
              f"{fmt(ea):>{col_ep}}  {str(be):>{col_bep}}  {fmt(gp):>{col_gap}}")


def print_deltas(experiments, baseline="001"):
    """打印变化量"""
    base = others = None
    for e in experiments:
        if baseline in e[0]:
            base = e[2]
        else:
            others = e
    if not base:
        print("\n⚠️  未找到 baseline")
        return
    bv, bp, be = base.get("best_val_loss", 0), base.get("perplexity", 0), base.get("epochs_actual", 0)
    W = min(TW, 100)
    ci, cv, cp, ce = 10, 10, 10, 8
    cd = max(10, W - ci - cv - cp - ce - 8)
    sep = "=" * W
    print(f"\n{sep}\n相对 baseline ({baseline}) 的变化（负数=变好，正数=变差）\n{sep}")
    print(f"{'ID':>{ci}}  {'描述':>{cd}}  {'Val变化':>{cv}}  {'PPL变化':>{cp}}  {'轮次变化':>{ce}}")
    print("-" * W)

    for eid, cfg, res in experiments:
        if baseline in eid:
            continue
        desc = trunc(cfg.get("description", ""), cd)
        v = res.get("best_val_loss")
        p = res.get("perplexity")
        e = res.get("epochs_actual")
        vd = f"{v - bv:+.2f}" if isinstance(v, (int, float)) else "?"
        pd = f"{p / bp:.1%}" if isinstance(p, (int, float)) and bp else "?"
        ed = f"{e - be:+d}" if isinstance(e, int) else "?"
        print(f"{eid:>{ci}}  {desc:>{cd}}  {vd:>{cv}}  {pd:>{cp}}  {ed:>{ce}}")


def print_generated(experiments):
    """打印生成文本"""
    W = min(TW, 100)
    print(f"\n{'=' * W}\n生成文本对比\n{'=' * W}")
    for eid, config, results in experiments:
        gen = results.get("generated", "")
        if gen:
            desc = config.get("description", "")[:40]
            print(f"\n  {eid} {desc}:\n    {gen[:W - 4]}")


def print_details(experiments):
    """打印某个实验的完整信息"""
    for eid, config, results in experiments:
        print(f"\n{'=' * 50}\n实验: {eid}\n{'=' * 50}")
        print("配置:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print("结果:")
        for k, v in results.items():
            print(f"  {k}: {v}")


# ── 交互式选择 ────────────────────────────

def interactive_select(experiments):
    while True:
        print(f"\n共 {len(experiments)} 个实验。选择查看方式：")
        print("  1) 全部显示")
        print("  2) 按标签筛选")
        print("  3) 按 ID 选择")
        print("  4) 只看最近 N 次")
        print("  5) 查看某个实验的完整配置和结果")
        print("  0) 退出")
        c = input("\n输入选项 (0-5): ").strip()
        if c == "0":
            return None
        elif c == "1":
            return experiments
        elif c == "2":
            all_tags = set()
            for _, cfg, _ in experiments:
                all_tags.update(cfg.get("tags", []))
            print(f"可用标签: {', '.join(sorted(all_tags))}")
            q = input("输入标签关键词: ").strip().lower()
            matched = [t for t in all_tags if q in t] if q else list(all_tags)
            if not matched:
                print("无匹配，显示全部。")
                return experiments
            flt = filter_experiments(experiments, tags=matched)
            print(f"匹配标签 {matched} → {len(flt)} 个")
            return flt if flt else experiments
        elif c == "3":
            q = input("输入关键词 (ID/描述/标签): ").strip()
            if not q:
                return experiments
            m = fuzzy_find(experiments, q)
            if not m:
                print(f"无匹配 '{q}'")
                continue
            if len(m) == 1:
                print(f"匹配: {m[0]}")
                return filter_experiments(experiments, ids=m)
            print(f"匹配 {len(m)} 个:")
            for i, eid in enumerate(m, 1):
                d = next((cfg.get("description", "")[:40] for e, cfg, _ in experiments if e == eid), "")
                print(f"  {i}) {eid} — {d}")
            s = input("选择编号 (回车=全部): ").strip()
            return filter_experiments(experiments, ids=[m[int(s) - 1]] if s.isdigit() and 1 <= int(s) <= len(m) else m)
        elif c == "4":
            n = input("看最近几次?: ").strip()
            return filter_experiments(experiments, last=int(n)) if n.isdigit() else experiments
        elif c == "5":
            print("可用实验:")
            for i, (eid, cfg, _) in enumerate(experiments, 1):
                print(f"  {i}) {eid} — {cfg.get('description', '')[:50]}")
            q = input("\n输入编号或关键词: ").strip()
            if not q:
                continue
            if q.isdigit():
                idx = int(q)
                if 1 <= idx <= len(experiments):
                    eid = experiments[idx - 1][0]
                else:
                    continue
            else:
                m = fuzzy_find(experiments, q)
                if not m:
                    continue
                eid = m[0] if len(m) == 1 else m[int(input("选择编号: ")) - 1]
            flt = filter_experiments(experiments, ids=[eid])
            if flt:
                print_details(flt)
            return None
        else:
            print("无效选项")


if __name__ == "__main__":
    exps = load_all()
    sel = interactive_select(exps)
    if sel is None:
        print("退出。")
        sys.exit(0)
    print_table(sel)
    print_deltas(sel)
    print_generated(sel)
