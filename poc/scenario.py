"""
场景生成模块：IEEE 39-bus N-1 contingency + 负荷扰动

场景筛选标准（保留）：
  - ACPF 收敛
  - 存在越限 (total_violations > 0)
  - 线路/变压器最大过载率 ∈ [105%, 125%]
  - 越限总数 ≤ 6
丢弃：
  - 不收敛 / 无越限 / 过载或越限数不在上述区间
"""
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import copy


def create_base_network():
    """创建 IEEE 39-bus 基础网络"""
    net = pn.case39()
    pp.runpp(net, algorithm='nr', max_iteration=30)
    return net


def classify_scenario(net):
    """
    对收敛的场景进行分类: 'keep' 或 'discard'

    Returns:
        category: 'keep' / 'discard' / 'no_violation'
        info: dict with violation details
    """
    v_pu = net.res_bus['vm_pu'].values

    n_v_low = int((v_pu < 0.94).sum())
    n_v_high = int((v_pu > 1.06).sum())

    line_loading = net.res_line['loading_percent'].values
    in_service_mask = net.line['in_service'].values
    active_loading = line_loading[in_service_mask]
    max_line_loading = float(active_loading.max()) if len(active_loading) > 0 else 0
    n_thermal = int((active_loading > 110).sum())  # N-1 紧急限值 110%

    trafo_loading = net.res_trafo['loading_percent'].values if len(net.trafo) > 0 else np.array([])
    n_trafo_thermal = int((trafo_loading > 110).sum()) if len(trafo_loading) > 0 else 0
    max_trafo_loading = float(trafo_loading.max()) if len(trafo_loading) > 0 else 0

    total_violations = n_v_low + n_v_high + n_thermal + n_trafo_thermal
    max_overload = max(max_line_loading, max_trafo_loading)

    info = {
        'total_violations': total_violations,
        'n_v_low': n_v_low,
        'n_v_high': n_v_high,
        'n_thermal': n_thermal + n_trafo_thermal,
        'max_overload_pct': max_overload,
    }

    if total_violations == 0:
        return 'no_violation', info

    # 保留: 过载 ∈ [105%, 125%] 且 越限 ≤ 6
    if 105 <= max_overload <= 125 and total_violations <= 6:
        return 'keep', info

    return 'discard', info


def generate_n1_scenarios(n_scenarios=20, seed=42):
    """
    生成 N-1 + 负荷扰动场景（仅保留合格场景）。

    Args:
        n_scenarios: 目标场景数
        seed: 随机种子

    Returns:
        scenarios: list of dicts
    """
    rng = np.random.RandomState(seed)

    master_net = create_base_network()
    n_lines = len(master_net.line)

    scenarios = []
    stats = {'attempts': 0, 'converged': 0, 'no_viol': 0, 'keep': 0, 'discard': 0}
    max_attempts = n_scenarios * 30

    while len(scenarios) < n_scenarios and stats['attempts'] < max_attempts:
        stats['attempts'] += 1
        net = copy.deepcopy(master_net)

        # N-1 断线 + 负荷 +5% ~ +30%
        line_idx = rng.randint(0, n_lines)
        net.line.at[line_idx, 'in_service'] = False
        removed_lines = [line_idx]
        global_scale = rng.uniform(1.05, 1.30)

        per_node_noise = rng.normal(1.0, 0.03, len(net.load))
        load_scale = global_scale * per_node_noise

        net.load['p_mw'] = master_net.load['p_mw'].values * load_scale
        net.load['q_mvar'] = master_net.load['q_mvar'].values * load_scale

        try:
            pp.runpp(net, algorithm='nr', max_iteration=50)
            if not net.converged:
                continue
        except Exception:
            continue

        stats['converged'] += 1

        category, info = classify_scenario(net)

        if category == 'no_violation':
            stats['no_viol'] += 1
            continue
        elif category == 'keep':
            stats['keep'] += 1
            scenarios.append({
                'net': net,
                'lines_removed': removed_lines,
                'load_scale_mean': float(np.mean(load_scale)),
                'category': 'N1',
                'info': info,
            })
        else:
            stats['discard'] += 1

    print(f"[Scenario Gen] 采样 {stats['attempts']} 次, "
          f"收敛 {stats['converged']}, "
          f"无越限 {stats['no_viol']}, "
          f"丢弃 {stats['discard']}")
    print(f"  → 保留: {len(scenarios)}/{n_scenarios} "
          f"(N-1, 过载 105-125%, 越限 ≤ 6)")

    if scenarios:
        max_ol = max(s['info']['max_overload_pct'] for s in scenarios)
        avg_viol = np.mean([s['info']['total_violations'] for s in scenarios])
        print(f"  统计: 最大过载={max_ol:.1f}%, 平均越限={avg_viol:.1f}处")

    return scenarios


if __name__ == '__main__':
    scen = generate_n1_scenarios(20)
    print("\n=== 场景 ===")
    for i, s in enumerate(scen):
        info = s['info']
        lines_str = ",".join(map(str, s.get('lines_removed', [])))
        print(f"  {i:2d}: L[{lines_str:4s}] | "
              f"负荷{s['load_scale_mean']:.2f}x | "
              f"越限{info['total_violations']:2d}处 | "
              f"最大过载{info['max_overload_pct']:.1f}%")
