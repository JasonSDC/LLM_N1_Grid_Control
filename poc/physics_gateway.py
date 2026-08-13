"""
物理网关模块：get_violations, g(x,z) 映射, Check-0, Check-1
"""
import pandapower as pp
import numpy as np
import copy
import math


# ────────────── 安全约束参数 ──────────────
V_MIN = 0.94   # p.u.
V_MAX = 1.06   # p.u.
THERMAL_LIMIT = 100.0   # % (正常)
THERMAL_LIMIT_EMG = 110.0  # % (N-1 紧急, τ_emg=15min)
LAMBDA_DAMP = 0.01      # 伪逆正则化系数
BETA_DAMP = 0.3         # 基础阻尼步长 (单机时用 0.3, 多机通过 adaptive_beta 放大)


def get_violations(net, emergency=True):
    """检测当前网络的越限情况"""
    result = {'voltage_low': [], 'voltage_high': [], 'thermal': [], 'total_count': 0}
    
    for bus_idx in net.res_bus.index:
        v = net.res_bus.at[bus_idx, 'vm_pu']
        if v < V_MIN:
            deficit = (V_MIN - v) / V_MIN * 100
            result['voltage_low'].append((bus_idx, v, deficit))
        elif v > V_MAX:
            excess = (v - V_MAX) / V_MAX * 100
            result['voltage_high'].append((bus_idx, v, excess))
    
    thermal_lim = THERMAL_LIMIT_EMG if emergency else THERMAL_LIMIT
    for line_idx in net.res_line.index:
        if not net.line.at[line_idx, 'in_service']:
            continue
        loading = net.res_line.at[line_idx, 'loading_percent']
        if loading > thermal_lim:
            excess = (loading - thermal_lim) / thermal_lim * 100
            result['thermal'].append(('line', line_idx, loading, excess))
    
    for trafo_idx in net.res_trafo.index:
        if not net.trafo.at[trafo_idx, 'in_service']:
            continue
        loading = net.res_trafo.at[trafo_idx, 'loading_percent']
        if loading > thermal_lim:
            excess = (loading - thermal_lim) / thermal_lim * 100
            result['thermal'].append(('trafo', trafo_idx, loading, excess))
    
    result['total_count'] = (len(result['voltage_low']) + 
                              len(result['voltage_high']) + 
                              len(result['thermal']))
    return result


def format_violations_str(violations):
    """将越限结果格式化为人可读字符串"""
    lines = []
    if violations['voltage_low']:
        lines.append("电压越下限:")
        for bus, v, deficit in sorted(violations['voltage_low'], key=lambda x: -x[2]):
            lines.append(f"  Bus {bus}: V={v:.4f} p.u. (低于下限 {deficit:.2f}%)")
    if violations['voltage_high']:
        lines.append("电压越上限:")
        for bus, v, excess in sorted(violations['voltage_high'], key=lambda x: -x[2]):
            lines.append(f"  Bus {bus}: V={v:.4f} p.u. (高于上限 {excess:.2f}%)")
    if violations['thermal']:
        lines.append("线路/变压器过载:")
        for etype, idx, loading, excess in sorted(violations['thermal'], key=lambda x: -x[3]):
            lines.append(f"  {etype} {idx}: loading={loading:.1f}% (excess {excess:.1f}%)")
    if not lines:
        lines.append("无越限")
    return "\n".join(lines)


def compute_sensitivity(net):
    """
    计算 AC 潮流灵敏度矩阵（数值微分）
    
    计算两类灵敏度：
    1. dV/dVset_g: 母线电压 对 发电机电压设定值 的灵敏度（电压控制）
    2. dLoading/dPg: 线路负载率 对 发电机有功出力 的灵敏度（潮流控制）
    """
    base_net = copy.deepcopy(net)
    try:
        pp.runpp(base_net, algorithm='nr', max_iteration=30)
    except Exception:
        pass
    
    gen_indices = base_net.gen.index.tolist()
    n_gen = len(gen_indices)
    n_bus = len(base_net.bus)
    n_line = len(base_net.line)
    
    base_v = base_net.res_bus['vm_pu'].values.copy()
    base_line_loading = base_net.res_line['loading_percent'].values.copy()
    
    delta_p = 1.0   # MW
    delta_v = 0.005  # p.u.
    
    dV_dVset = np.zeros((n_bus, n_gen))
    dV_dPg = np.zeros((n_bus, n_gen))
    dLoading_dPg = np.zeros((n_line, n_gen))
    
    for g_i, gen_idx in enumerate(gen_indices):
        perturbed = copy.deepcopy(base_net)
        perturbed.gen.at[gen_idx, 'vm_pu'] += delta_v
        try:
            pp.runpp(perturbed, algorithm='nr', max_iteration=30)
            if perturbed.converged:
                dV_dVset[:, g_i] = (perturbed.res_bus['vm_pu'].values - base_v) / delta_v
        except Exception:
            pass
        
        perturbed2 = copy.deepcopy(base_net)
        perturbed2.gen.at[gen_idx, 'p_mw'] += delta_p
        try:
            pp.runpp(perturbed2, algorithm='nr', max_iteration=30)
            if perturbed2.converged:
                dV_dPg[:, g_i] = (perturbed2.res_bus['vm_pu'].values - base_v) / delta_p
                dLoading_dPg[:, g_i] = (perturbed2.res_line['loading_percent'].values - base_line_loading) / delta_p
        except Exception:
            pass
    
    return {
        'gen_indices': gen_indices,
        'dV_dVset': dV_dVset,
        'dV_dPg': dV_dPg,
        'dLoading_dPg': dLoading_dPg,
        'base_v': base_v,
        'base_loading': base_line_loading,
    }


def intent_to_action(net, intent, sensitivity=None, max_iter=10, verbose=False):
    """
    g(x_k, z): 意图 → 连续动作映射 (SLP + 全局目标自动接管)
    
    LLM 只需选机 (generators)，SLP 自动扫描全网越限:
      电压: J_v = dV_dVset, 低压升 / 高压降
      热过载: J_p @ ΔP ≈ -excess（仅线路）
    
    每一步:
      1. get_violations(work_net) → 自动发现所有越限
      2. 电压用 dV_dVset 更新 ΔVset；热用 dLoading_dPg 更新 ΔP
      3. clip 到容量边界, ACPF 刷新, 重复直到越限清零或无进步
    
    Returns:
        dict: {'gen_idx': {'dp': MW, 'dv': p.u.}, ...}
    """
    gen_indices = intent['generators']
    
    if not gen_indices:
        return {}
    
    # ── 初始化累积动作 ──
    cumulative = {}
    for g in gen_indices:
        if g in net.gen.index:
            cumulative[g] = {'dp': 0.0, 'dv': 0.0}
    
    if not cumulative:
        return {}
    
    # ── 动态自适应步长 ──
    num_gens = len(cumulative)
    adaptive_beta = min(0.8, 0.3 * math.sqrt(num_gens))
    if verbose:
        print(f"    [SLP] 自适应参数: num_gens={num_gens}, β={adaptive_beta:.2f}, max_iter={max_iter}")
    
    # ── 工作网络 ──
    work_net = copy.deepcopy(net)
    
    for iteration in range(max_iter):
        # ─── Step 1: 灵敏度 ───
        if iteration == 0 and sensitivity is not None:
            sens = sensitivity
        else:
            sens = compute_sensitivity(work_net)
        
        all_gen = sens['gen_indices']
        gen_col_indices = [all_gen.index(g) for g in gen_indices if g in all_gen]
        
        if not gen_col_indices:
            break
        
        # ─── Step 2: 全局越限自动检测 ───
        current_viols = get_violations(work_net)
        
        if current_viols['total_count'] == 0:
            if verbose:
                print(f"    [SLP] iter {iteration}: ✅ 越限清零!")
            break
        
        # 快照：ACPF 失败时回退本轮对 cumulative 的修改
        prev_cumulative = {g: dict(d) for g, d in cumulative.items()}
        any_step_taken = False
        
        # ─── Step 2a: 电压控制 (全网所有电压越限) ───
        v_low_list = current_viols.get('voltage_low', [])
        v_high_list = current_viols.get('voltage_high', [])
        
        if v_low_list or v_high_list:
            e_v = []
            v_bus_list = []
            
            for bus, v, deficit in v_low_list:
                e_v.append((V_MIN - v) / V_MIN)      # 正: 需要升压
                v_bus_list.append(bus)
            for bus, v, excess in v_high_list:
                e_v.append(-(v - V_MAX) / V_MAX)      # 负: 需要降压
                v_bus_list.append(bus)
            
            if any(abs(e) > 1e-4 for e in e_v):
                e = np.array(e_v).reshape(-1, 1)
                K_v = 3.0
                e = e * K_v
                
                J_rows = []
                for bus in v_bus_list:
                    row = sens['dV_dVset'][bus, gen_col_indices]
                    J_rows.append(row)
                
                J_z = np.array(J_rows)
                v_dim, c_dim = J_z.shape
                
                JJT = J_z @ J_z.T + LAMBDA_DAMP * np.eye(v_dim)
                try:
                    delta_u = adaptive_beta * J_z.T @ np.linalg.solve(JJT, e)
                except np.linalg.LinAlgError:
                    delta_u = np.zeros((c_dim, 1))
                
                for i, g_col in enumerate(gen_col_indices):
                    gen_idx = all_gen[g_col]
                    dv_step = float(delta_u[i, 0])
                    
                    orig_v = net.gen.at[gen_idx, 'vm_pu']
                    new_cumul_dv = cumulative[gen_idx]['dv'] + dv_step
                    new_cumul_dv = np.clip(new_cumul_dv, 0.94 - orig_v, 1.10 - orig_v)
                    
                    if abs(new_cumul_dv - cumulative[gen_idx]['dv']) > 1e-5:
                        any_step_taken = True
                    cumulative[gen_idx]['dv'] = new_cumul_dv
        
        # ─── Step 2b: 热过载控制 (仅线路, trafo 无灵敏度数据) ───
        # 目标: J_z @ ΔP ≈ -excess，使负载率下降
        thermal_list = [t for t in current_viols.get('thermal', []) if t[0] == 'line']
        
        if thermal_list:
            e_t = []
            t_idx_list = []
            
            for etype, idx, loading, excess in thermal_list:
                e_t.append((loading - THERMAL_LIMIT_EMG) / THERMAL_LIMIT_EMG)
                t_idx_list.append(idx)
            
            if any(e > 1e-4 for e in e_t):
                e = -np.array(e_t).reshape(-1, 1)  # 负号: 降低过载
                total_gen_capacity = sum(work_net.gen['max_p_mw'].values)
                K_p = total_gen_capacity * 0.3
                e = e * K_p
                
                J_rows = []
                for idx in t_idx_list:
                    row = sens['dLoading_dPg'][idx, gen_col_indices] / 100.0
                    J_rows.append(row)
                
                J_z = np.array(J_rows)
                v_dim, c_dim = J_z.shape
                
                JJT = J_z @ J_z.T + LAMBDA_DAMP * np.eye(v_dim)
                try:
                    delta_u = adaptive_beta * J_z.T @ np.linalg.solve(JJT, e)
                except np.linalg.LinAlgError:
                    delta_u = np.zeros((c_dim, 1))
                
                for i, g_col in enumerate(gen_col_indices):
                    gen_idx = all_gen[g_col]
                    dp_step = float(delta_u[i, 0])
                    
                    orig_p = net.gen.at[gen_idx, 'p_mw']
                    p_max = net.gen.at[gen_idx, 'max_p_mw']
                    p_min = net.gen.at[gen_idx, 'min_p_mw']
                    new_cumul_dp = cumulative[gen_idx]['dp'] + dp_step
                    new_cumul_dp = np.clip(new_cumul_dp, p_min - orig_p, p_max - orig_p)
                    
                    if abs(new_cumul_dp - cumulative[gen_idx]['dp']) > 0.01:
                        any_step_taken = True
                    cumulative[gen_idx]['dp'] = new_cumul_dp
        
        # ─── Step 3: 无进步 → 退出 ───
        if not any_step_taken:
            if verbose:
                print(f"    [SLP] iter {iteration}: 无进步, 提前退出")
            break
        
        # ─── Step 4: ACPF 刷新 ───
        work_net = copy.deepcopy(net)
        for gen_idx, deltas in cumulative.items():
            work_net.gen.at[gen_idx, 'p_mw'] = net.gen.at[gen_idx, 'p_mw'] + deltas['dp']
            work_net.gen.at[gen_idx, 'vm_pu'] = net.gen.at[gen_idx, 'vm_pu'] + deltas['dv']
        
        try:
            pp.runpp(work_net, algorithm='nr', max_iteration=30)
            if not work_net.converged:
                cumulative = prev_cumulative
                if verbose:
                    print(f"    [SLP] iter {iteration}: ACPF 不收敛, 回退本轮并退出")
                break
        except Exception:
            cumulative = prev_cumulative
            if verbose:
                print(f"    [SLP] iter {iteration}: ACPF 异常, 回退本轮并退出")
            break
        
        # ─── Step 5: 日志 ───
        remaining_viols = get_violations(work_net)
        if verbose:
            print(f"    [SLP] iter {iteration}: 残余越限 {remaining_viols['total_count']} 处, "
                  f"ΣΔP={sum(abs(d['dp']) for d in cumulative.values()):.1f} MW, "
                  f"ΣΔV={sum(abs(d['dv']) for d in cumulative.values()):.4f} p.u.")
    
    # ── 过滤零变化 ──
    actions = {}
    for gen_idx, deltas in cumulative.items():
        if abs(deltas['dp']) > 0.01 or abs(deltas['dv']) > 1e-5:
            actions[gen_idx] = deltas
    
    return actions




def check_0(actions, net):
    """Check-0: 容量边界检查 O(1)"""
    violated = []
    for gen_idx, deltas in actions.items():
        dp = deltas.get('dp', 0)
        dv = deltas.get('dv', 0)
        
        current_p = net.gen.at[gen_idx, 'p_mw']
        new_p = current_p + dp
        p_max = net.gen.at[gen_idx, 'max_p_mw']
        p_min = net.gen.at[gen_idx, 'min_p_mw']
        if new_p > p_max * 1.01 or new_p < p_min * 0.99:
            violated.append((gen_idx, 'p', new_p, p_min, p_max))
        
        current_v = net.gen.at[gen_idx, 'vm_pu']
        new_v = current_v + dv
        # 与 SLP clip / LLM prompt 一致: Vset ∈ [0.94, 1.10]
        if new_v > 1.10 or new_v < 0.94:
            violated.append((gen_idx, 'v', new_v, 0.94, 1.10))
    
    return len(violated) == 0, violated


def check_1(net, actions):
    """Check-1: 施加动作后运行 ACPF，检查所有约束"""
    new_net = copy.deepcopy(net)
    
    for gen_idx, deltas in actions.items():
        dp = deltas.get('dp', 0)
        dv = deltas.get('dv', 0)
        new_net.gen.at[gen_idx, 'p_mw'] += dp
        new_net.gen.at[gen_idx, 'vm_pu'] += dv
    
    try:
        pp.runpp(new_net, algorithm='nr', max_iteration=50)
        if not new_net.converged:
            return False, {'total_count': -1, 'note': 'ACPF 不收敛'}, new_net
    except Exception as exc:
        return False, {'total_count': -1, 'note': f'ACPF 异常: {exc}'}, new_net
    
    new_violations = get_violations(new_net)
    ok = new_violations['total_count'] == 0
    return ok, new_violations, new_net


def apply_fallback(net, violations):
    """
    Fallback 策略 π_base: 保守安全兜底。
    
    策略：迭代式联合调节
    1. 每轮将所有发电机电压设定值向 1.0 p.u. 收缩
    2. 每轮切 5% 负荷（受 max_shedding = 20% 总负荷 上限约束）
    3. 最多 20 轮
    
    诊断：详细打印每一步的失败原因（不收敛 vs 仍有越限）
    """
    new_net = copy.deepcopy(net)
    total_shed = 0.0
    
    # ── 切负荷上限保护 ──
    original_total_load = net.load['p_mw'].sum()
    max_shedding = 0.20 * original_total_load
    shed_exhausted = False
    
    n_not_converge = 0
    n_still_violated = 0
    last_violation_count = violations.get('total_count', 0)
    
    for step in range(20):
        # ─── 电压调节：所有发电机设定值向 1.0 收缩 ───
        for gen_idx in new_net.gen.index:
            current_v = new_net.gen.at[gen_idx, 'vm_pu']
            new_net.gen.at[gen_idx, 'vm_pu'] = current_v + 0.20 * (1.0 - current_v)
        
        # ─── 负荷削减：每步切 5%，受上限约束 ───
        if not shed_exhausted:
            shed_ratio = 0.05
            current_load_total = new_net.load['p_mw'].sum()
            shed_this_step = current_load_total * shed_ratio
            
            if total_shed + shed_this_step > max_shedding:
                shed_this_step = max(0, max_shedding - total_shed)
                if shed_this_step > 0:
                    actual_ratio = shed_this_step / current_load_total
                    new_net.load['p_mw'] *= (1 - actual_ratio)
                    new_net.load['q_mvar'] *= (1 - actual_ratio)
                    total_shed += shed_this_step
                shed_exhausted = True
                print(f"    [Fallback] Step {step}: 切负荷达上限 {max_shedding:.1f} MW (原始负荷的20%)")
            else:
                new_net.load['p_mw'] *= (1 - shed_ratio)
                new_net.load['q_mvar'] *= (1 - shed_ratio)
                total_shed += shed_this_step
        
        # ─── ACPF 检查 + 诊断 ───
        try:
            pp.runpp(new_net, algorithm='nr', max_iteration=100)
            if not new_net.converged:
                n_not_converge += 1
                print(f"    [Fallback] Step {step}: ❌ ACPF 不收敛 "
                      f"(已切 {total_shed:.1f} MW = {100*total_shed/original_total_load:.1f}%)")
                continue
            
            v = get_violations(new_net)
            if v['total_count'] == 0:
                print(f"    [Fallback] Step {step}: ✅ 成功! 切负荷 {total_shed:.1f} MW "
                      f"({100*total_shed/original_total_load:.1f}%)")
                return True, total_shed, new_net
            else:
                n_still_violated += 1
                last_violation_count = v['total_count']
                if step % 5 == 0:
                    print(f"    [Fallback] Step {step}: 仍有 {v['total_count']} 处越限 "
                          f"(已切 {total_shed:.1f} MW = {100*total_shed/original_total_load:.1f}%)")
        except Exception as e:
            n_not_converge += 1
            print(f"    [Fallback] Step {step}: ❌ ACPF 异常: {e}")
            continue
    
    # ─── 失败诊断 ───
    print(f"    [Fallback] 💀 死因诊断: {n_not_converge}次不收敛, "
          f"{n_still_violated}次仍越限(最后{last_violation_count}处), "
          f"共切负荷{total_shed:.1f}MW ({100*total_shed/original_total_load:.1f}%), "
          f"上限{max_shedding:.1f}MW ({'已达上限' if shed_exhausted else '未达上限'})")
    
    return False, total_shed, new_net
