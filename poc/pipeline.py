"""
Algorithm 1 主循环：Deadline-Aware RA Pipeline
"""
import time
import copy
from poc.physics_gateway import (
    get_violations, intent_to_action, check_0, check_1, 
    apply_fallback, compute_sensitivity, format_violations_str
)
from poc.intent_engine import (
    rule_based_intent, llm_intent, generate_sensitivity_hint
)
from poc.math_solver import (solve_with_slsqp, solve_with_slsqp_sparse, solve_with_opf, 
                             solve_with_slsqp_topk, solve_with_opf_l1, solve_with_opf_voltage_only)

def run_pipeline(net, mode='rule', api_key=None, t_max=30.0, verbose=True):
    """
    运行 Algorithm 1 完整流程
    """
    start_time = time.time()
    result = {
        'success': False, 'fallback': False, 'method': 'none',
        'total_time_ms': 0, 'llm_time_ms': 0, 'n_llm_calls': 0,
        'initial_violations': 0, 'final_violations': 0,
        'actions': {}, 'shed_mw': 0,
        'cost_dp': 0.0, 'cost_dv': 0.0, 'cost_total': 0.0, 'n_act': 0,
        'sensitivity_time_ms': 0.0,
    }
    
    # ──── Step 1: get_violations ────
    violations = get_violations(net)
    result['initial_violations'] = violations['total_count']
    
    if violations['total_count'] == 0:
        if verbose:
            print("  [Pipeline] 无越限，无需控制")
        result['success'] = True
        result['method'] = 'no_action'
        return result
    
    if verbose:
        print(f"  [Pipeline] 检测到 {violations['total_count']} 处越限")
        print(f"    {format_violations_str(violations)}")
    
    # ──── 计算灵敏度 (OPF 系列不需要，跳过以避免浪费 20 次 ACPF) ────
    if mode not in ['opf', 'opf_l1', 'opf_vonly']:
        t_sens_start = time.time()
        sensitivity = compute_sensitivity(net)
        result['sensitivity_time_ms'] = (time.time() - t_sens_start) * 1000
    else:
        sensitivity = None
        result['sensitivity_time_ms'] = 0.0
    
    max_retries = 5  # 初次 + 4 次重试 (保证最多 5 轮)
    
    # ──── Semantic A: 固定原始目标 ────
    target_violations = copy.deepcopy(violations)
    target_viol_list = []
    target_viol_list += [('voltage', v[0]) for v in violations.get('voltage_low', [])]
    target_viol_list += [('voltage', v[0]) for v in violations.get('voltage_high', [])]
    target_viol_list += [('thermal', v[1]) for v in violations.get('thermal', [])]
    
    hint = None
    base_hint = ""
    if sensitivity is not None:
        base_hint = generate_sensitivity_hint(net, target_viol_list, sensitivity) or ""
    
    for attempt in range(max_retries):
        # 检查时限
        t_elapsed = time.time() - start_time
        if t_elapsed >= t_max:
            if verbose:
                print(f"  [Pipeline] 超时 ({t_elapsed:.1f}s)，触发 fallback")
            return _do_fallback(net, target_violations, result, start_time, verbose)
        
        # ──── Step 1.5: 数学极值解法 (Math/OPF Baseline) ────
        if mode in ['math', 'math_sparse', 'math_topk', 'opf', 'opf_l1', 'opf_vonly']:
            if attempt > 0:
                if verbose: print(f"  [{mode.upper()}] 不支持带历史反馈重试，直接 Fallback")
                return _do_fallback(net, target_violations, result, start_time, verbose)
                
            if mode == 'math':
                actions, math_ms = solve_with_slsqp(net, target_violations, sensitivity, verbose)
            elif mode == 'math_sparse':
                actions, math_ms = solve_with_slsqp_sparse(net, target_violations, sensitivity, verbose, sparse_budget=2)
            elif mode == 'math_topk':
                actions, math_ms = solve_with_slsqp_topk(net, target_violations, sensitivity, verbose, topk_budget=2)
            elif mode == 'opf_l1':
                actions, opf_ms = solve_with_opf_l1(net, verbose)
            elif mode == 'opf_vonly':
                actions, opf_ms = solve_with_opf_voltage_only(net, verbose)
            else:
                actions, opf_ms = solve_with_opf(net, verbose)
                
            result['actions'] = actions
            if not actions:
                return _do_fallback(net, target_violations, result, start_time, verbose)
                
            # 直达 Check-0 物理门禁，不需要过 LLM 意图
            pass
            
        # ──── Step 2: 生成意图 (LLM/Rule) ────
        elif mode == 'llm' and api_key:
            intent, llm_ms, raw = llm_intent(
                target_violations, net, api_key, hint=hint, sensitivity=sensitivity
            )
            result['llm_time_ms'] += llm_ms
            result['n_llm_calls'] += 1
            if verbose:
                print(f"  [LLM] 延迟={llm_ms:.0f}ms, 意图={'有效' if intent else '无效'}")
            if intent is None:
                if verbose:
                    print(f"  [LLM] 意图解析失败，回退到规则引擎")
                intent = rule_based_intent(target_violations, net, sensitivity)
        elif mode == 'rule':
            # 规则选机确定性：Check 失败后重试不会改变 generators，直接 Fallback
            if attempt > 0:
                if verbose:
                    print(f"  [Rule] 确定性意图无重试收益，直接 Fallback")
                return _do_fallback(net, target_violations, result, start_time, verbose)
            intent = rule_based_intent(target_violations, net, sensitivity)
        else:
            intent = None  # 安全回退
            
        if mode in ['llm', 'rule']:
            if intent is None:
                if verbose:
                    print(f"  [Pipeline] 无法生成意图 ({mode})，触发 fallback")
                return _do_fallback(net, target_violations, result, start_time, verbose)
            
            if verbose:
                print(f"  [Intent] 来源={intent['source']}, "
                      f"发电机={intent['generators']}, "
                      f"目标={len(intent['targets'])}个越限")
            
            # ──── Step 3: g(x_k, z) 映射 (SLP 内部迭代) ────
            actions = intent_to_action(net, intent, sensitivity, verbose=verbose)
            result['actions'] = actions
            
            if not actions:
                if verbose:
                    print("  [Pipeline] g(x,z) 映射返回空动作")
                continue
        
        if verbose:
            for g, deltas in actions.items():
                parts = []
                if abs(deltas.get('dp', 0)) > 0.01:
                    parts.append(f"ΔP={deltas['dp']:+.2f} MW")
                if abs(deltas.get('dv', 0)) > 0.001:
                    parts.append(f"ΔV={deltas['dv']:+.4f} p.u.")
                if parts:
                    print(f"    Gen {g}: {', '.join(parts)}")
        
        # ──── Step 4: Check-0 ────
        ok_0, viol_0 = check_0(actions, net)
        if not ok_0:
            if verbose:
                print(f"  [Check-0] ❌ 容量越界: {viol_0}")
            fail_msg = f"\n[Retry Feedback - Attempt {attempt+1}]: ❌ Check-0 (Physical Limits) FAILED. Your proposed generators {list(actions.keys())} caused out-of-bounds capacity errors: {viol_0}. You MUST select DIFFERENT generators or change your target strategy."
            hint = base_hint + fail_msg
            continue  # 重试
        
        if verbose:
            print(f"  [Check-0] ✅ 通过")
        
        # ──── Step 5: Check-1 (Static ACPF Feasibility) ────
        ok_1, new_viols, new_net = check_1(net, actions)
        if verbose:
            if ok_1:
                print(f"  [Check-1] ✅ 通过! 稳态越限清零")
            else:
                remaining = new_viols.get('total_count', '?')
                print(f"  [Check-1] ❌ 控制后仍有 {remaining} 处稳态越限")
        
        if not ok_1:
            rem = new_viols.get('total_count', '?')
            if 'voltage_low' in new_viols:
                detail_str = format_violations_str(new_viols)
                fail_msg = f"\n[Retry Feedback - Attempt {attempt+1}]: ❌ Check-1 (Steady-State AC Flow) FAILED. Your selection {list(actions.keys())} was structurally insufficient. {rem} violations STILL REMAIN:\n{detail_str}\n\nYou MUST select DIFFERENT generators with higher sensitivities, or increase the number of targeted generators."
            else:
                note = new_viols.get('note', 'ACPF convergence failure')
                fail_msg = f"\n[Retry Feedback - Attempt {attempt+1}]: ❌ Check-1 (Steady-State AC Flow) FAILED. Your selection {list(actions.keys())} caused extreme stress: {note}. You MUST use a more CONSERVATIVE strategy (smaller voltage/power changes) or different generators."
            hint = base_hint + fail_msg
            continue

        # Check-0、Check-1 均通过
        # ── 计算标准化控制成本 (Evaluation Metric) ──
        from poc.metrics import evaluate_action_cost
        cost_info = evaluate_action_cost(actions)
        result.update(cost_info)
        
        result['success'] = True
        result['method'] = 'intent'
        result['final_violations'] = 0
        result['total_time_ms'] = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"  [控制成本] Σ|ΔP|={cost_info['cost_dp']:.1f} MW, "
                  f"Σ|ΔV|={cost_info['cost_dv']:.4f} p.u., "
                  f"n_act={cost_info['n_act']}, "
                  f"Total={cost_info['cost_total']:.1f}")
        return result
    
    # 重试耗尽
    return _do_fallback(net, target_violations, result, start_time, verbose)


def _do_fallback(net, violations, result, start_time, verbose):
    """执行 fallback"""
    if verbose:
        print("  [Fallback] 触发 fallback...")
    
    ok, shed_mw, new_net = apply_fallback(net, violations)
    result['fallback'] = True
    result['shed_mw'] = shed_mw
    result['method'] = 'fallback'
    result['total_time_ms'] = (time.time() - start_time) * 1000
    
    if ok:
        result['success'] = True
        result['final_violations'] = 0
        if verbose:
            print(f"  [Fallback] ✅ 成功, 切负荷 {shed_mw:.1f} MW")
    else:
        result['success'] = False
        post_viols = get_violations(new_net) if new_net.converged else {'total_count': -1}
        result['final_violations'] = post_viols.get('total_count', -1)
        if verbose:
            print(f"  [Fallback] ❌ 仍有 {result['final_violations']} 处越限")
    
    return result
