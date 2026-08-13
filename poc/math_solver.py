import numpy as np
from scipy.optimize import minimize
import time
import pandapower as pp
import copy
def solve_with_slsqp(net, violations, sensitivity, verbose=True, lambda_v=100.0):
    """
    使用传统数值优化 (SLSQP / 内点法思想) 求解越限消除问题。
    基于全局 Jacobian (Sensitivity) 构建线性化不等式约束。
    
    决策变量 x: [dv_0, dp_0, dv_1, dp_1, ..., dv_9, dp_9]
    长度 = 2 * n_gen (IEEE 39 为 20)
    
    目标函数: min \sum |dp_i| + lambda_v * \sum |dv_i|
    """
    return _solve_slsqp_core(net, violations, sensitivity, verbose, sparse_budget=None, lambda_v=lambda_v)


def solve_with_slsqp_sparse(net, violations, sensitivity, verbose=True, sparse_budget=3, lambda_v=100.0):
    """
    B2: Sparse Classical Optimizer
    完全随机截取前几个发电机来凑数，仅用于验证数量本身的影响。
    """
    return _solve_slsqp_core(net, violations, sensitivity, verbose, sparse_budget=sparse_budget, lambda_v=lambda_v)


def solve_with_slsqp_topk(net, violations, sensitivity, verbose=True, topk_budget=3, lambda_v=100.0):
    """
    B5: Top-K Sensitivity + Sparse Optimizer
    计算所有越限元素绝对灵敏度最大的 Top-K 台机组作为全部动作空间。
    用于对标 LLM 的选点直觉。
    """
    return _solve_slsqp_core(net, violations, sensitivity, verbose, topk_budget=topk_budget, lambda_v=lambda_v)


def _solve_slsqp_core(net, violations, sensitivity, verbose=True, sparse_budget=None, topk_budget=None, lambda_v=100.0):
    """
    核心 SLSQP 求解器容器，支持可选的稀疏限制
    """
    t0 = time.time()
    gen_idx = sensitivity['gen_indices']
    n_gen = len(gen_idx)
    n_vars = 2 * n_gen
    
    # 提取越限项
    v_low = violations.get('voltage_low', [])
    v_high = violations.get('voltage_high', [])
    thermal = violations.get('thermal', [])
    
    if len(v_low) + len(v_high) + len(thermal) == 0:
        return {}, 0.0
        
    # 构建约束条件 list
    # minimize 方法要求不等式约束为: f(x) >= 0
    constraints = []
    
    # 1. 消除低压越限: V_current + dV_dVset @ dv_vector >= 0.94
    for bus, v_curr, _ in v_low:
        def make_vlow_cons(b, vc):
            return lambda x: vc + np.sum(sensitivity['dV_dVset'][b, :] * x[0::2]) - 0.9405  # 加微小裕度
        constraints.append({'type': 'ineq', 'fun': make_vlow_cons(bus, v_curr)})
        
    # 2. 消除高压越限: V_current + dV_dVset @ dv_vector <= 1.06  => 1.06 - (...) >= 0
    for bus, v_curr, _ in v_high:
        def make_vhigh_cons(b, vc):
            return lambda x: 1.0595 - (vc + np.sum(sensitivity['dV_dVset'][b, :] * x[0::2]))
        constraints.append({'type': 'ineq', 'fun': make_vhigh_cons(bus, v_curr)})
        
    # 3. 消除热过载 (仅限线路上，通过改变 P): Loading_curr + dLoading_dPg @ dp_vector <= 110
    # => 110 - (...) >= 0
    THERMAL_LIMIT = 109.9  # 加微小裕度
    for etype, idx, load_curr, _ in thermal:
        if etype == 'line':
            def make_thermal_cons(i, lc):
                # 灵敏度的单位是 % / MW
                return lambda x: THERMAL_LIMIT - (lc + np.sum(sensitivity['dLoading_dPg'][i, :] * x[1::2]))
            constraints.append({'type': 'ineq', 'fun': make_thermal_cons(idx, load_curr)})
            
    # 4. 构建变量物理边界 Bounds 和 稀疏度遮罩
    bounds = []
    
    # 构建 Top-K 候选列表
    candidate_scores = {g: 0.0 for g in gen_idx}
    
    for bus, v_curr, _ in v_low + v_high:
        sens_row = np.abs(sensitivity['dV_dVset'][bus, :])
        for i, val in enumerate(sens_row):
            candidate_scores[gen_idx[i]] += val
            
    for etype, idx, _, _ in thermal:
        if etype == 'line':
            sens_row = np.abs(sensitivity['dLoading_dPg'][idx, :])
            # normalize the thermal sensitivity relative to voltage to prevent dominance
            for i, val in enumerate(sens_row):
                candidate_scores[gen_idx[i]] += val * 0.01 
    
    # 对所有发电机得分进行排序，选出全局最敏感的前 k 台
    sorted_gens = sorted(candidate_scores.keys(), key=lambda k: candidate_scores[k], reverse=True)
    
    target_budget = topk_budget if topk_budget is not None else sparse_budget
    allowed_gens = set()
    
    if target_budget is not None and len(gen_idx) > target_budget:
        if topk_budget is not None:
            # B5: 精准选取全局绝对灵敏度最高的 k 台机组
            allowed_gens = set(sorted_gens[:topk_budget])
            if verbose: print(f"  [Math Top-K] 物理灵敏度截断 k={topk_budget}, 开放执行器: {allowed_gens}")
        else:
            # B2: 退化为从 top 列表中随机或者按列截取，以产生与 top-k 不同的一般化组合
            # 由于之前是简单的 append 覆盖，这里为了纯粹对标"降维但不精准"的情况
            # 采用非最优的随机发电机截断
            import random
            allowed_gens = set(random.sample(list(gen_idx), sparse_budget))
            if verbose: print(f"  [Math Sparse] 随机截断 k={sparse_budget}, 开放执行器: {allowed_gens}")

    for g in gen_idx:
        # 如果开启了稀疏且当前发电机不在允许列表内，强行死锁动作
        if target_budget is not None and g not in allowed_gens:
            bounds.append((0.0, 0.0))  # dv 锁死
            bounds.append((0.0, 0.0))  # dp 锁死
            continue
            
        # dv bounds限制: 不能超出 0.94~1.10 的机端物理限值
        v_curr = net.gen.at[g, 'vm_pu']
        dv_min = 0.94 - v_curr
        dv_max = 1.10 - v_curr
        # 限制单次调节幅度，防止过度超调导致非线性误差过大
        dv_min = max(dv_min, -0.05)
        dv_max = min(dv_max,  0.05)
        
        # dp bounds: 不能超过 P_min, P_max
        p_curr = net.gen.at[g, 'p_mw']
        p_min = net.gen.at[g, 'min_p_mw']
        p_max = net.gen.at[g, 'max_p_mw']
        # 缺失值处理: 设置极宽的保护限值
        if np.isnan(p_min): p_min = 0.0
        if np.isnan(p_max): p_max = 2000.0
        dp_min = p_min - p_curr
        dp_max = p_max - p_curr
        
        bounds.append((dv_min, dv_max))
        bounds.append((dp_min, dp_max))
        
    # 目标函数 (带有平滑 L1 范数以辅助梯度下降)
    def objective(x):
        dv_vec = x[0::2]
        dp_vec = x[1::2]
        # 使用 pseudo-Huber 或 sqrt(x^2 + epsilon) 来实现平滑的绝对值函数，避免梯度不连续
        eps = 1e-4
        cost_dv = np.sum(np.sqrt(dv_vec**2 + eps)) * lambda_v
        cost_dp = np.sum(np.sqrt(dp_vec**2 + eps))
        return cost_dv + cost_dp

    # 初始猜测
    x0 = np.zeros(n_vars)
    
    # 求解 SLSQP
    options = {'maxiter': 100, 'ftol': 1e-4, 'disp': False}
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options=options)
    except Exception as e:
        if verbose: print(f"  [Math] SLSQP 运行崩溃: {e}")
        return {}, (time.time() - t0) * 1000
        
    t_ms = (time.time() - t0) * 1000
    
    if not res.success:
        if verbose: print(f"  [Math] SLSQP 优化失败 ({t_ms:.0f}ms): {res.message}")
        return {}, t_ms
        
    # 组装返回的 actions 格式
    actions = {}
    x_opt = res.x
    for i, g in enumerate(gen_idx):
        dv = x_opt[2*i]
        dp = x_opt[2*i + 1]
        
        # 过滤极小动作噪声
        if abs(dv) < 1e-4 and abs(dp) < 1e-2:
            continue
            
        actions[g] = {'dv': float(dv), 'dp': float(dp)}
        
    if verbose:
        from poc.metrics import evaluate_action_cost
        eval_metrics = evaluate_action_cost(actions, lambda_v=lambda_v)
        prefix = "[Math Sparse]" if sparse_budget is not None else "[Math]"
        print(f"  {prefix} SLSQP 求解成功 ({t_ms:.0f}ms, Cost={eval_metrics['cost_total']:.1f})")
        
    return actions, t_ms

def solve_with_opf(net, verbose=True):
    """
    L2-OPF: min sum (P_i - P_i0)^2 (Post-Contingency Corrective OPF).
    使用 Pandapower 内置的 AC OPF (pypower interior point).
    """
    t0 = time.time()
    net_opf = _setup_opf_network(net)
    
    # L2 cost: C(P) = P^2 - 2*P0*P ≡ (P - P0)^2 + const
    for idx, gen in net_opf.gen.iterrows():
        p0 = gen['p_mw']
        if np.isnan(p0): p0 = 0.0
        pp.create_poly_cost(net_opf, idx, et='gen', cp1_eur_per_mw=-2.0 * p0, cp2_eur_per_mw2=1.0)
        
    # ext_grid cost + 固定电压
    for idx, ext in net_opf.ext_grid.iterrows():
        try:
            p0 = net.res_ext_grid.at[idx, 'p_mw']
            if np.isnan(p0): p0 = 0.0
        except Exception:
            p0 = 0.0
        pp.create_poly_cost(net_opf, idx, et='ext_grid', cp1_eur_per_mw=-2.0 * p0, cp2_eur_per_mw2=1.0)
        
        # 固定 ext_grid 电压不参与优化
        bus_idx = ext['bus']
        v0 = ext['vm_pu']
        net_opf.bus.at[bus_idx, 'min_vm_pu'] = v0 - 1e-4
        net_opf.bus.at[bus_idx, 'max_vm_pu'] = v0 + 1e-4

    try:
        pp.runopp(net_opf, verbose=False, calculate_voltage_angles=True)
    except Exception as e:
        if verbose: print(f"  [OPF] AC OPF 运行崩溃: {e}")
        return {}, (time.time() - t0) * 1000
    
    t_ms = (time.time() - t0) * 1000
    if not net_opf.get('OPF_converged', False):
        if verbose: print(f"  [OPF] AC OPF 不收敛 ({t_ms:.0f}ms)")
        return {}, t_ms
        
    actions = _extract_opf_actions(net, net_opf)
    
    if verbose:
        from poc.metrics import evaluate_action_cost
        eval_metrics = evaluate_action_cost(actions)
        print(f"  [OPF] L2-OPF 求解成功 ({t_ms:.0f}ms, Cost≈{eval_metrics['cost_total']:.1f})")
        
    return actions, t_ms


def _setup_opf_network(net):
    """
    OPF 网络初始化的公共逻辑：设置 bus limits, thermal limits, gen/ext_grid bounds.
    返回 deepcopy 后的网络。
    """
    net_opf = copy.deepcopy(net)
    
    # Bus 电压上下限
    if 'min_vm_pu' not in net_opf.bus.columns:
        net_opf.bus['min_vm_pu'] = 0.9405
    else:
        net_opf.bus['min_vm_pu'] = net_opf.bus['min_vm_pu'].fillna(0.9405)
        
    if 'max_vm_pu' not in net_opf.bus.columns:
        net_opf.bus['max_vm_pu'] = 1.0595
    else:
        net_opf.bus['max_vm_pu'] = net_opf.bus['max_vm_pu'].fillna(1.0595)
        
    # 线路和变压器热稳定极限
    net_opf.line['max_loading_percent'] = 109.9
    net_opf.trafo['max_loading_percent'] = 109.9
    
    # Gen bounds: 填充 NaN 为合理默认值
    for idx in net_opf.gen.index:
        if 'min_p_mw' in net_opf.gen.columns and np.isnan(net_opf.gen.at[idx, 'min_p_mw']):
            net_opf.gen.at[idx, 'min_p_mw'] = 0.0
        if 'max_p_mw' in net_opf.gen.columns and np.isnan(net_opf.gen.at[idx, 'max_p_mw']):
            net_opf.gen.at[idx, 'max_p_mw'] = net.gen.at[idx, 'p_mw'] * 2.0
        if 'min_q_mvar' in net_opf.gen.columns and np.isnan(net_opf.gen.at[idx, 'min_q_mvar']):
            net_opf.gen.at[idx, 'min_q_mvar'] = -300.0
        if 'max_q_mvar' in net_opf.gen.columns and np.isnan(net_opf.gen.at[idx, 'max_q_mvar']):
            net_opf.gen.at[idx, 'max_q_mvar'] = 300.0
        
    # ext_grid bounds
    for idx in net_opf.ext_grid.index:
        if 'min_q_mvar' in net_opf.ext_grid.columns:
            net_opf.ext_grid.at[idx, 'min_q_mvar'] = -9999.0
        if 'max_q_mvar' in net_opf.ext_grid.columns:
            net_opf.ext_grid.at[idx, 'max_q_mvar'] = 9999.0
        if 'min_p_mw' in net_opf.ext_grid.columns:
            net_opf.ext_grid.at[idx, 'min_p_mw'] = -9999.0
        if 'max_p_mw' in net_opf.ext_grid.columns:
            net_opf.ext_grid.at[idx, 'max_p_mw'] = 9999.0
    
    # 清理自带 cost
    net_opf.poly_cost.drop(net_opf.poly_cost.index, inplace=True)
    net_opf.pwl_cost.drop(net_opf.pwl_cost.index, inplace=True)
    
    return net_opf


def _extract_opf_actions(net, net_opf, lock_p=False):
    """
    从收敛的 OPF 网络中提取统一格式的 actions dict.
    lock_p=True 时跳过 ΔP 提取（ΔV-only 模式）.
    """
    actions = {}
    for idx, gen in net.gen.iterrows():
        dp = 0.0 if lock_p else (net_opf.res_gen.at[idx, 'p_mw'] - gen['p_mw'])
        dv = net_opf.res_bus.at[gen['bus'], 'vm_pu'] - gen['vm_pu']
        if abs(dp) > 1e-4 or abs(dv) > 1e-4:
            actions[idx] = {'dp': float(dp), 'dv': float(dv)}
    return actions


def solve_with_opf_l1(net, verbose=True):
    """
    L1-OPF: 近似 L1 目标的 AC Optimal Power Flow.
    
    目标函数: min Σ|ΔP_i| ≈ min Σ 0.001·(P_i - P_i0)²
    
    使用极小二次项 cp2=0.001 + 线性项 cp1=-2·0.001·P0 (quadratic centered at P0).
    极小 cp2 产生近似 L1 的稀疏效果: 只在必须调整时才动 P, 调整量不会被"分散".
    与 L2-OPF (cp2=1.0) 的区别: L2 的强二次惩罚导致"平均分散", 
    而这里的弱二次惩罚让 OPF 更倾向于集中调整少数发电机.
    
    ΔV 惩罚通过 bus voltage bounds 收紧隐式实现：
    将每个 gen bus 的 vm 允许范围限制在当前值 ±0.08 p.u.，
    这样 OPF 不会无代价地大幅调整电压。
    """
    t0 = time.time()
    net_opf = _setup_opf_network(net)
    
    # ─── Near-L1 cost: 极小二次项 centered at P0 ───
    # C(P) = 0.001·(P - P0)² = 0.001·P² - 0.002·P0·P + const
    # 极小 cp2 → Hessian 可正定但几乎平坦 → 稀疏解
    CP2_REG = 0.001  # 正则化系数
    for idx, gen in net_opf.gen.iterrows():
        p0 = gen['p_mw']
        if np.isnan(p0): p0 = 0.0
        pp.create_poly_cost(net_opf, idx, et='gen',
                           cp1_eur_per_mw=-2.0 * CP2_REG * p0,
                           cp2_eur_per_mw2=CP2_REG)
        
        # 收紧 gen bus 电压 bounds，隐式惩罚 ΔV
        # 允许 ±0.08 p.u. 调节空间，但不允许无代价大幅偏离
        bus_idx = gen['bus']
        v0 = gen['vm_pu']
        # 不超出全局安全限值
        net_opf.bus.at[bus_idx, 'min_vm_pu'] = max(0.9405, v0 - 0.08)
        net_opf.bus.at[bus_idx, 'max_vm_pu'] = min(1.0595, v0 + 0.08)
    
    # ext_grid cost + 固定电压
    for idx, ext in net_opf.ext_grid.iterrows():
        try:
            p0 = net.res_ext_grid.at[idx, 'p_mw']
            if np.isnan(p0): p0 = 0.0
        except Exception:
            p0 = 0.0
        pp.create_poly_cost(net_opf, idx, et='ext_grid',
                           cp1_eur_per_mw=-2.0 * CP2_REG * p0,
                           cp2_eur_per_mw2=CP2_REG)
        
        bus_idx = ext['bus']
        v0 = ext['vm_pu']
        net_opf.bus.at[bus_idx, 'min_vm_pu'] = v0 - 1e-4
        net_opf.bus.at[bus_idx, 'max_vm_pu'] = v0 + 1e-4

    try:
        pp.runopp(net_opf, verbose=False, calculate_voltage_angles=True)
    except Exception as e:
        if verbose: print(f"  [OPF-L1] AC OPF (L1) 运行崩溃: {e}")
        return {}, (time.time() - t0) * 1000
    
    t_ms = (time.time() - t0) * 1000
    if not net_opf.get('OPF_converged', False):
        if verbose: print(f"  [OPF-L1] AC OPF (L1) 不收敛 ({t_ms:.0f}ms)")
        return {}, t_ms
        
    actions = _extract_opf_actions(net, net_opf)
    
    if verbose:
        from poc.metrics import evaluate_action_cost
        eval_metrics = evaluate_action_cost(actions)
        print(f"  [OPF-L1] L1-OPF 求解成功 ({t_ms:.0f}ms, Cost={eval_metrics['cost_total']:.1f})")
        
    return actions, t_ms


def solve_with_opf_voltage_only(net, verbose=True):
    """
    ΔV-only OPF: 锁死所有发电机有功出力，仅允许电压设定值调节.
    
    直接对标 LLM-SLP 的"纯电压控制"策略:
    - 所有 gen: min_p_mw = max_p_mw = p_mw (有功完全锁死)
    - 只通过 bus voltage bounds 内的 vm_pu 自由度消除越限
    - cost 仅为极小正则化项，防止 Hessian 奇异
    
    注意: 此模式只能消除电压越限，无法消除热过载.
    如果场景中有热过载，该模式大概率会 fallback.
    """
    t0 = time.time()
    net_opf = _setup_opf_network(net)
    
    # ─── 锁死有功: min_p = max_p = p_current ───
    for idx in net_opf.gen.index:
        p_curr = net_opf.gen.at[idx, 'p_mw']
        if np.isnan(p_curr): p_curr = 0.0
        net_opf.gen.at[idx, 'min_p_mw'] = p_curr
        net_opf.gen.at[idx, 'max_p_mw'] = p_curr
    
    # ext_grid 有功也收紧（允许微小容差避免不可行）
    for idx in net_opf.ext_grid.index:
        try:
            p0 = net.res_ext_grid.at[idx, 'p_mw']
            if np.isnan(p0): p0 = 0.0
        except Exception:
            p0 = 0.0
        net_opf.ext_grid.at[idx, 'min_p_mw'] = p0 - 5.0  # 微小容差, 松弛节点需要平衡
        net_opf.ext_grid.at[idx, 'max_p_mw'] = p0 + 5.0
    
    # ─── 极小 cost (纯正则化, 不驱动有功变化) ───
    for idx, gen in net_opf.gen.iterrows():
        p0 = gen['p_mw']
        if np.isnan(p0): p0 = 0.0
        pp.create_poly_cost(net_opf, idx, et='gen',
                           cp1_eur_per_mw=0.0,
                           cp2_eur_per_mw2=0.001)
    
    for idx, ext in net_opf.ext_grid.iterrows():
        pp.create_poly_cost(net_opf, idx, et='ext_grid',
                           cp1_eur_per_mw=0.0,
                           cp2_eur_per_mw2=0.001)
        
        # 固定 ext_grid 电压
        bus_idx = ext['bus']
        v0 = ext['vm_pu']
        net_opf.bus.at[bus_idx, 'min_vm_pu'] = v0 - 1e-4
        net_opf.bus.at[bus_idx, 'max_vm_pu'] = v0 + 1e-4

    try:
        pp.runopp(net_opf, verbose=False, calculate_voltage_angles=True)
    except Exception as e:
        if verbose: print(f"  [OPF-Vonly] ΔV-only OPF 运行崩溃: {e}")
        return {}, (time.time() - t0) * 1000
    
    t_ms = (time.time() - t0) * 1000
    if not net_opf.get('OPF_converged', False):
        if verbose: print(f"  [OPF-Vonly] ΔV-only OPF 不收敛 ({t_ms:.0f}ms)")
        return {}, t_ms
        
    # lock_p=True: 强制 ΔP=0，只提取 ΔV
    actions = _extract_opf_actions(net, net_opf, lock_p=True)
    
    if verbose:
        from poc.metrics import evaluate_action_cost
        eval_metrics = evaluate_action_cost(actions)
        print(f"  [OPF-Vonly] ΔV-only OPF 求解成功 ({t_ms:.0f}ms, Cost={eval_metrics['cost_total']:.1f})")
        
    return actions, t_ms
