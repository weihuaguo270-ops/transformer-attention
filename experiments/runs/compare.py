"""
实验对比查看器 — 一键看所有实验的关键指标和变化

用法:
  python -m experiments.runs.compare
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RUNS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all():
    """加载所有实验"""
    experiments = []
    for d in sorted(os.listdir(RUNS_DIR)):
        if not d[0].isdigit():
            continue
        config_path = os.path.join(RUNS_DIR, d, "config.json")
        results_path = os.path.join(RUNS_DIR, d, "results.json")
        if not os.path.exists(config_path) or not os.path.exists(results_path):
            continue
        with open(config_path) as f:
            config = json.load(f)
        with open(results_path) as f:
            results = json.load(f)
        experiments.append((d, config, results))
    return experiments


def print_table(experiments):
    """打印关键指标对比表"""
    print("=" * 90)
    print(f"{'ID':>8} {'名称':>16} {'模型':>20} {'Val':>8} {'PPL':>8} "
          f"{'轮次':>6} {'最佳轮':>6} {'Train-Val差距':>12}")
    print("-" * 90)

    for exp_id, config, results in experiments:
        name = config.get("desc", exp_id)[:16]
        d_model = config.get("d_model", "?")
        lr = config.get("lr", "?")
        model_str = f"d={d_model} lr={lr}"

        best_val = results.get("best_val_loss", "?")
        ppl = results.get("perplexity", "?")
        epochs_actual = results.get("epochs_actual", "?")
        best_epoch = results.get("best_epoch", "?")
        train_loss = results.get("final_train_loss", "?")
        val_loss = results.get("final_val_loss", "?")
        gap = round(val_loss - train_loss, 2) if isinstance(train_loss, (int, float)) and isinstance(val_loss, (int, float)) else "?"

        ppl_str = f"{ppl:.0f}" if isinstance(ppl, (int, float)) else f"{ppl}"
        best_ep = f"{best_epoch}" if best_epoch else "-"

        print(f"{exp_id:>8} {name:>16} {model_str:>20} "
              f"{best_val:>8} {ppl_str:>8} {epochs_actual:>6} {best_ep:>6} {gap:>12}")


def print_deltas(experiments):
    """打印相对于 baseline (001) 的变化"""
    baseline_id = None
    baseline_results = None
    for exp_id, config, results in experiments:
        if "001" in exp_id:
            baseline_id = exp_id
            baseline_results = results
            break

    if not baseline_results:
        print("\n⚠️  未找到 baseline (001)，无法计算变化")
        return

    print(f"\n{'=' * 90}")
    print(f"相对于 {baseline_id} 的变化（正数=变差，负数=变好）")
    print(f"{'=' * 90}")
    print(f"{'ID':>8} {'名称':>16} {'Val变化':>10} {'PPL变化':>10} {'轮次变化':>10}")
    print("-" * 60)

    b_val = baseline_results.get("best_val_loss", 0)
    b_ppl = baseline_results.get("perplexity", 0)
    b_ep = baseline_results.get("epochs_actual", 0)

    for exp_id, config, results in experiments:
        if "001" in exp_id:
            continue

        name = config.get("desc", exp_id)[:16]
        val = results.get("best_val_loss", "?")
        ppl = results.get("perplexity", "?")
        ep = results.get("epochs_actual", "?")

        val_delta = f"{val - b_val:+.2f}" if isinstance(val, (int, float)) else "?"
        ppl_delta = f"{ppl / b_ppl:.1%}" if isinstance(ppl, (int, float)) and b_ppl else "?"
        ep_delta = f"{ep - b_ep:+d}" if isinstance(ep, int) else "?"

        print(f"{exp_id:>8} {name:>16} {val_delta:>10} {ppl_delta:>10} {ep_delta:>10}")


def print_generated(experiments):
    """打印所有实验的生成文本"""
    print(f"\n{'=' * 90}")
    print("生成文本对比")
    print(f"{'=' * 90}")
    for exp_id, config, results in experiments:
        gen = results.get("generated", "")
        if gen:
            print(f"\n  {exp_id} {config.get('desc', '').strip()}:")
            print(f"    {gen[:120]}")


if __name__ == "__main__":
    experiments = load_all()
    if not experiments:
        print("没有找到实验记录。")
        sys.exit(1)

    print_table(experiments)
    print_deltas(experiments)
    print_generated(experiments)
