"""
PoC 主运行脚本：多基线对比实验
  - AC-OPF (L2, 原始)
  - AC-OPF-L1 (近 L1 目标, 公平对标评估指标)
  - ΔV-only OPF (锁死有功, 对标 LLM-SLP 控制范式)
  - Rule+SLP (B6 消融: 规则引擎 + SLP 映射)
  - LLM-RA (我们的方法: Claude + SLP + Check-0/1)
"""
import os
import numpy as np
from poc.scenario import generate_n1_scenarios
from poc.pipeline import run_pipeline

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def run_experiment(mode, scenarios, api_key=None):
    """对一组场景运行 pipeline"""
    results = []
    for i, s in enumerate(scenarios):
        info = s['info']
        print(f"\n{'='*60}")
        lines_str = ",".join(map(str, s.get('lines_removed', [s.get('line_removed')])))
        print(f"场景 {i+1}/{len(scenarios)} | 断线 L[{lines_str}] | "
              f"负荷 {s['load_scale_mean']:.2f}x | "
              f"越限 {info['total_violations']}处 | "
              f"最大过载 {info['max_overload_pct']:.0f}% | "
              f"类型={s['category']} | 模式={mode}")
        print(f"{'='*60}")

        r = run_pipeline(
            s['net'], mode=mode, api_key=api_key,
            t_max=30.0, verbose=True,
        )
        results.append(r)
    return results


def print_summary(label, results):
    """打印汇总统计"""
    total = len(results)
    if total == 0:
        print(f"\n[{label}] 无场景")
        return

    intent_ok = sum(1 for r in results if r['method'] == 'intent')
    fb_ok = sum(1 for r in results if r['fallback'] and r['success'])
    fb_fail = sum(1 for r in results if r['fallback'] and not r['success'])
    overall_ok = sum(1 for r in results if r['success'])

    avg_time = np.mean([r['total_time_ms'] for r in results])
    avg_llm = np.mean([r['llm_time_ms'] for r in results]) if any(r['llm_time_ms'] > 0 for r in results) else 0
    total_llm_calls = sum(r['n_llm_calls'] for r in results)
    total_shed = sum(r['shed_mw'] for r in results)
    avg_sens_time = np.mean([r.get('sensitivity_time_ms', 0) for r in results])

    intent_results = [r for r in results if r['method'] == 'intent']

    print(f"\n{'='*60}")
    print(f"  结果汇总:  {label}")
    print(f"{'='*60}")
    print(f"  总场景数:        {total}")
    print(f"  ──────────────────────────────")
    print(f"  意图成功 (无Fallback): {intent_ok}/{total} ({100*intent_ok/total:.1f}%)")
    print(f"  Fallback 成功:        {fb_ok}/{total} ({100*fb_ok/total:.1f}%)")
    print(f"  Fallback 失败:        {fb_fail}/{total} ({100*fb_fail/total:.1f}%)")
    print(f"  ──────────────────────────────")
    print(f"  整体成功率:      {overall_ok}/{total} ({100*overall_ok/total:.1f}%)")
    print(f"  ──────────────────────────────")
    print(f"  平均延迟 (total):  {avg_time:.0f} ms")
    print(f"  平均灵敏度计算:    {avg_sens_time:.0f} ms")
    if avg_llm > 0:
        print(f"  平均 LLM 延迟:   {avg_llm:.0f} ms")
        print(f"  LLM 总调用次数:  {total_llm_calls}")
    print(f"  总切负荷:        {total_shed:.1f} MW")
    if intent_results:
        avg_cost_dp = np.mean([r['cost_dp'] for r in intent_results])
        avg_cost_dv = np.mean([r['cost_dv'] for r in intent_results])
        avg_cost_total = np.mean([r['cost_total'] for r in intent_results])
        avg_n_act = np.mean([r['n_act'] for r in intent_results])
        print(f"  ──────── 控制成本 (仅意图成功) ────────")
        print(f"  平均 Σ|ΔP|:     {avg_cost_dp:.1f} MW")
        print(f"  平均 Σ|ΔV|:     {avg_cost_dv:.4f} p.u.")
        print(f"  平均 Actuators: {avg_n_act:.1f}")
        print(f"  平均 Total Cost: {avg_cost_total:.1f}")
    print(f"{'='*60}")


METHOD_LABELS = {
    'opf':       'AC-OPF(L2)',
    'opf_l1':    'AC-OPF(L1)',
    'opf_vonly': 'ΔV-only OPF',
    'rule':      'Rule+SLP',
    'llm':       'LLM-RA',
}


def _is_intent_success(r):
    return r is not None and r.get('method') == 'intent' and not r.get('fallback')


def print_comparison(results_dict):
    """打印多方法对比表"""

    def status_str(r):
        if r is None:
            return '   N/A     '
        if _is_intent_success(r):
            return '✅ 意图成功 '
        elif r['fallback'] and r['success']:
            return '⚠️ Fallback✓'
        else:
            return '❌ Fallback✗'

    active_methods = [m for m in ['opf', 'opf_l1', 'opf_vonly', 'rule', 'llm']
                      if m in results_dict and len(results_dict[m]) > 0]
    if len(active_methods) < 2:
        return

    n_scenarios = max(len(results_dict[m]) for m in active_methods)

    print(f"\n{'='*120}")
    method_names = ' vs '.join(METHOD_LABELS.get(m, m) for m in active_methods)
    print(f"  对比矩阵: {method_names} --- {n_scenarios} 个场景")
    print(f"{'='*120}")

    header = f"{'场景':>4s} |"
    for m in active_methods:
        lbl = METHOD_LABELS.get(m, m)
        header += f" [{lbl:^14s}] {'Cost':>8s} |"
    header += f" {'Best':>10s}"
    print(header)
    print("-" * (len(header) + 10))

    wins = {m: 0 for m in active_methods}

    for i in range(n_scenarios):
        row = f"  {i:<4d} |"
        costs = {}

        for m in active_methods:
            r = results_dict[m][i] if i < len(results_dict[m]) else None
            s = status_str(r)

            if r and _is_intent_success(r):
                c = r['cost_total']
                cs = f"{c:>8.1f}"
                costs[m] = c
            else:
                cs = "     N/A"

            row += f" {s} {cs} |"

        if costs:
            best_method = min(costs, key=costs.get)
            best_cost = costs[best_method]
            others = {m: c for m, c in costs.items() if m != best_method}
            if others:
                second_best = min(others.values())
                if best_cost < second_best * 0.95:
                    winner = METHOD_LABELS.get(best_method, best_method)
                    wins[best_method] += 1
                else:
                    winner = "Tie"
            else:
                winner = METHOD_LABELS.get(best_method, best_method) + " (唯一)"
                wins[best_method] += 1
        else:
            winner = "---"

        row += f" {winner:>14s}"
        print(row)

    print(f"\n  {'─'*60}")
    win_str = ", ".join(f"{METHOD_LABELS.get(m, m)}={wins[m]}" for m in active_methods)
    print(f"  🏆 胜场统计: {win_str}")

    key_pairs = [
        ('opf', 'llm'),
        ('opf_l1', 'llm'),
        ('opf_vonly', 'llm'),
        ('rule', 'llm'),
    ]

    for m1, m2 in key_pairs:
        if m1 not in active_methods or m2 not in active_methods:
            continue

        ok1 = {i for i in range(len(results_dict[m1])) if _is_intent_success(results_dict[m1][i])}
        ok2 = {i for i in range(len(results_dict[m2])) if _is_intent_success(results_dict[m2][i])}
        both = sorted(ok1 & ok2)

        l1 = METHOD_LABELS.get(m1, m1)
        l2 = METHOD_LABELS.get(m2, m2)

        if both:
            avg_c1 = np.mean([results_dict[m1][i]['cost_total'] for i in both])
            avg_c2 = np.mean([results_dict[m2][i]['cost_total'] for i in both])
            avg_dp1 = np.mean([results_dict[m1][i]['cost_dp'] for i in both])
            avg_dp2 = np.mean([results_dict[m2][i]['cost_dp'] for i in both])
            avg_dv1 = np.mean([results_dict[m1][i]['cost_dv'] for i in both])
            avg_dv2 = np.mean([results_dict[m2][i]['cost_dv'] for i in both])
            avg_nact1 = np.mean([results_dict[m1][i]['n_act'] for i in both])
            avg_nact2 = np.mean([results_dict[m2][i]['n_act'] for i in both])

            print(f"\n  📊 {l1} vs {l2} 交集对比 ({len(both)} 个共同成功场景):")
            print(f"      J_eval:    {l1}={avg_c1:.1f}  vs  {l2}={avg_c2:.1f}")
            print(f"      Σ|ΔP|:    {l1}={avg_dp1:.1f}MW  vs  {l2}={avg_dp2:.1f}MW")
            print(f"      Σ|ΔV|:    {l1}={avg_dv1:.4f}pu  vs  {l2}={avg_dv2:.4f}pu")
            print(f"      n_act:     {l1}={avg_nact1:.1f}  vs  {l2}={avg_nact2:.1f}")

        excl1 = sorted(ok1 - ok2)
        excl2 = sorted(ok2 - ok1)
        if excl2:
            print(f"  🌟 只有 {l2} 能解决: 场景 {excl2}")
        if excl1:
            print(f"  📌 只有 {l1} 能解决: 场景 {excl1}")


def main():
    print("=" * 60)
    print("  LLM-RA PoC: Multi-Baseline Comparison")
    print("  IEEE 39-bus | N-1 + load disturbance")
    print("  Methods: AC-OPF(L2), AC-OPF(L1), ΔV-only OPF,")
    print("           Rule+SLP (B6 Ablation), LLM-RA")
    print("=" * 60)

    if not CLAUDE_API_KEY:
        print("⚠️ ANTHROPIC_API_KEY 未设置. LLM-RA 实验将被跳过.")
        print("  设置方法: export ANTHROPIC_API_KEY='sk-ant-...'")

    print("[DEBUG] 开始生成 N-1 测试场景...")
    scenarios = generate_n1_scenarios(n_scenarios=20, seed=42)
    print("[DEBUG] 场景生成完毕！")

    results = {}

    print("\n\n" + "=" * 60)
    print("  Baseline 1: AC-OPF (L2, Post-Contingency Corrective OPF)")
    print("=" * 60)
    results['opf'] = run_experiment('opf', scenarios)
    print_summary("AC-OPF(L2)", results['opf'])

    print("\n\n" + "=" * 60)
    print("  Baseline 2: AC-OPF (L1, Near-Linear Cost)")
    print("=" * 60)
    results['opf_l1'] = run_experiment('opf_l1', scenarios)
    print_summary("AC-OPF(L1)", results['opf_l1'])

    print("\n\n" + "=" * 60)
    print("  Baseline 3: ΔV-only OPF (Lock P, Voltage Control Only)")
    print("=" * 60)
    results['opf_vonly'] = run_experiment('opf_vonly', scenarios)
    print_summary("ΔV-only OPF", results['opf_vonly'])

    print("\n\n" + "=" * 60)
    print("  Baseline 4 (B6 Ablation): Rule+SLP")
    print("=" * 60)
    results['rule'] = run_experiment('rule', scenarios)
    print_summary("Rule+SLP", results['rule'])

    if CLAUDE_API_KEY:
        print("\n\n" + "=" * 60)
        print("  Ours: LLM-RA + Check-0/1")
        print("=" * 60)
        results['llm'] = run_experiment('llm', scenarios, api_key=CLAUDE_API_KEY)
        print_summary("LLM-RA", results['llm'])
    else:
        print("\n⚠️ 跳过 LLM-RA 实验 (无 API key)")

    print_comparison(results)


if __name__ == '__main__':
    main()
