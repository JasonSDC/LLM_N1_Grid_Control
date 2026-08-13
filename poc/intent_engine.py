"""
意图引擎模块：规则引擎 (B6) + Claude LLM 意图生成
"""
import json
import time
import numpy as np


def rule_based_intent(violations, net, sensitivity):
    """
    B6 规则引擎：灵敏度最大匹配，分电压/热类型处理
    
    每个违限选灵敏度最高的 top-2 台发电机, 最多覆盖 top-5 违限,
    总共不超过 5 台独立发电机 (保持稀疏性).
    """
    all_viols = []
    
    for bus, v, deficit in violations.get('voltage_low', []):
        all_viols.append(('voltage', bus, 'up', deficit))
    for bus, v, excess in violations.get('voltage_high', []):
        all_viols.append(('voltage', bus, 'down', excess))
    for etype, idx, loading, excess in violations.get('thermal', []):
        if etype == 'line':
            all_viols.append(('thermal', idx, 'reduce', excess))
    
    all_viols.sort(key=lambda x: -x[3])
    
    if not all_viols:
        return None
    
    # 覆盖 top-5 违限 (比原来的 3 更充分)
    top_n_viols = min(5, len(all_viols))
    targets = []
    selected_gens = set()
    gen_indices = sensitivity['gen_indices']
    MAX_GENS = 5  # 总发电机数量上限, 保持稀疏
    
    for i in range(top_n_viols):
        viol_type, idx, direction, severity = all_viols[i]
        
        if viol_type == 'voltage':
            sens_row = np.abs(sensitivity['dV_dVset'][idx, :])
        elif viol_type == 'thermal':
            sens_row = np.abs(sensitivity['dLoading_dPg'][idx, :])
        else:
            continue
        
        # Top-2: 选对该约束灵敏度最高的 2 台发电机
        top_n = min(2, len(gen_indices))
        top_gen_cols = np.argsort(-sens_row)[:top_n]
        for g_col in top_gen_cols:
            if len(selected_gens) < MAX_GENS:
                selected_gens.add(gen_indices[g_col])
        
        targets.append({
            'type': viol_type,
            'idx': idx,
            'direction': direction,
        })
    
    if not selected_gens:
        return None
    
    return {
        'generators': list(selected_gens),
        'targets': targets,
        'action_type': 'auto',
        'source': 'rule_engine',
    }


def parse_llm_intent(response_text, net):
    """解析 LLM 返回的 JSON 意图"""
    try:
        text = response_text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        data = json.loads(text)
        
        generators = data.get('generators', [])
        targets_raw = data.get('targets', [])
        
        valid_gens = [g for g in generators if g in net.gen.index.tolist()]
        
        targets = []
        for t in targets_raw:
            targets.append({
                'type': t.get('type', 'voltage'),
                'idx': int(t.get('idx', 0)),
                'direction': t.get('direction', 'down'),
            })
        
        if not valid_gens or not targets:
            return None
        
        return {
            'generators': valid_gens,
            'targets': targets,
            'action_type': data.get('action_type', 'auto'),
            'source': 'llm',
        }
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        return None


def llm_intent(violations, net, api_key, hint=None, sensitivity=None):

    try:
        import anthropic
    except ImportError:
        print("[LLM] anthropic 未安装")
        return None, 0, ""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # ──── 发电机信息（含当前运行点和可调范围）────
    gen_info = []
    for gen_idx in net.gen.index:
        current_p = float(net.gen.at[gen_idx, 'p_mw'])
        max_p = float(net.gen.at[gen_idx, 'max_p_mw'])
        min_p = float(net.gen.at[gen_idx, 'min_p_mw'])
        gen_info.append({
            'gen_idx': int(gen_idx),
            'bus': int(net.gen.at[gen_idx, 'bus']),
            'p_mw': round(current_p, 1),
            'adjustable_range_mw': f"[{round(min_p - current_p, 1)}, +{round(max_p - current_p, 1)}]",
            'vm_pu': round(float(net.gen.at[gen_idx, 'vm_pu']), 4),
        })
    
    # ──── 构建灵敏度映射表 (结构化 JSON) ────
    sensitivity_map = []
    if sensitivity is not None:
        gen_indices = sensitivity['gen_indices']
        
        for bus, v, deficit in violations.get('voltage_low', []):
            sens_row = sensitivity['dV_dVset'][bus, :]
            top_k = min(3, len(gen_indices))
            top_cols = np.argsort(-np.abs(sens_row))[:top_k]
            top_gens = [
                {"gen_idx": int(gen_indices[c]),
                 "bus": int(net.gen.at[gen_indices[c], 'bus']),
                 "sensitivity_dV_dVset": round(float(sens_row[c]), 4)}
                for c in top_cols if abs(sens_row[c]) > 0.01
            ]
            if top_gens:
                sensitivity_map.append({
                    "violation": f"UNDER-VOLTAGE Bus {bus}",
                    "top_sensitive_generators": top_gens
                })
        
        for bus, v, excess in violations.get('voltage_high', []):
            sens_row = sensitivity['dV_dVset'][bus, :]
            top_k = min(3, len(gen_indices))
            top_cols = np.argsort(-np.abs(sens_row))[:top_k]
            top_gens = [
                {"gen_idx": int(gen_indices[c]),
                 "bus": int(net.gen.at[gen_indices[c], 'bus']),
                 "sensitivity_dV_dVset": round(float(sens_row[c]), 4)}
                for c in top_cols if abs(sens_row[c]) > 0.01
            ]
            if top_gens:
                sensitivity_map.append({
                    "violation": f"OVER-VOLTAGE Bus {bus}",
                    "top_sensitive_generators": top_gens
                })
        
        for etype, idx, loading, excess in violations.get('thermal', []):
            if etype == 'line':
                sens_row = sensitivity['dLoading_dPg'][idx, :]
                top_k = min(3, len(gen_indices))
                top_cols = np.argsort(-np.abs(sens_row))[:top_k]
                top_gens = [
                    {"gen_idx": int(gen_indices[c]),
                     "bus": int(net.gen.at[gen_indices[c], 'bus']),
                     "sensitivity_dLoading_dPg": round(float(sens_row[c]), 3)}
                    for c in top_cols if abs(sens_row[c]) > 0.01
                ]
                if top_gens:
                    sensitivity_map.append({
                        "violation": f"OVERLOAD Line {idx}",
                        "top_sensitive_generators": top_gens
                    })
    
    # ──── Violation summary: 展示全部越限 ────
    all_viols_flat = []
    
    for bus, v, deficit in violations.get('voltage_low', []):
        all_viols_flat.append((deficit, f"[UNDER-VOLTAGE] Bus {bus}: V={v:.4f} p.u., {deficit:.1f}% below limit 0.94"))
    for bus, v, excess in violations.get('voltage_high', []):
        all_viols_flat.append((excess, f"[OVER-VOLTAGE] Bus {bus}: V={v:.4f} p.u., {excess:.1f}% above limit 1.06"))
    for etype, idx, loading, excess in violations.get('thermal', []):
        all_viols_flat.append((excess, f"[OVERLOAD] {etype} {idx}: loading={loading:.0f}%, {excess:.0f}% over limit"))
        
    all_viols_flat.sort(key=lambda x: -x[0])
    total_v = violations.get('total_count', 0)
    viol_summary = "\n".join(item[1] for item in all_viols_flat)
    
    # ──── 构建 system_prompt (灵敏度表放在固定位置) ────
    sensitivity_section = ""
    if sensitivity_map:
        sensitivity_section = f"""\n
**Sensitivity Analysis** (Physics-based ranking of most effective generators per violation):
{json.dumps(sensitivity_map, indent=2)}

**IMPORTANT**: Use the sensitivity data above to guide your generator selection. Higher absolute sensitivity means the generator has stronger physical influence on that violation. Prefer generators that appear across MULTIPLE violations."""
    
    system_prompt = f"""You are a power system emergency control agent for an IEEE 39-bus network.

**Your task**: Select generators to eliminate the most critical violations. Think about which generators can BEST address each violation type simultaneously without creating new problems.
If multiple violations exist, you MUST ensure your generator selection and strategy can address ALL of them simultaneously. Do NOT just target one violation if there are multiple.

**Generator Selection Rule**:
- You should normally use 1-3 generators. Using more than 3 generators is rarely necessary.
- **CRITICAL**: You MUST use as FEW generators as possible! Selecting fewer generators reduces control cost and operational complexity.

**Hard Physical Constraints** (MUST NOT violate):
- Generator active power P must stay within [P_min, P_max]
- Generator voltage setpoint V must stay within [0.94, 1.10] p.u.
- If a generator's current vm_pu is already near 1.10, do NOT select it for voltage boost!
- Check each generator's adjustable_range_mw before selecting it

**Control Strategy**:
- Under-voltage → Select generators with HIGH sensitivity_dV_dVset for the violated bus, raise their voltage setpoint
- Over-voltage → Select generators with HIGH sensitivity_dV_dVset for the violated bus, lower their voltage setpoint  
- Line overload → Select generators with HIGH sensitivity_dLoading_dPg that can redistribute power flow away from the overloaded line
- When multiple violation types coexist, prefer generators that can address BOTH without conflict

**Available Generators** ({len(gen_info)} units):
{json.dumps(gen_info, indent=2)}{sensitivity_section}

**Output ONLY valid JSON, no other text**:
{{"generators": [idx1, idx2], "targets": [{{"type": "voltage"/"thermal", "idx": bus_or_line_id, "direction": "up"/"down"/"reduce"}}, {{"type": "...", ...}}], "action_type": "auto", "reasoning": "brief reason"}}"""

    user_prompt = f"Current network has {total_v} total violations. You MUST address ALL of them:\n{viol_summary}"
    if hint:
        user_prompt += f"\n\n[Physics Guidance & Feedback]:\n{hint}"
    
    t0 = time.time()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0,
        )
        raw = response.content[0].text
        latency = (time.time() - t0) * 1000
        
        intent = parse_llm_intent(raw, net)
        if intent is None:
            print(f"  [LLM DEBUG] 解析失败, 原始响应前200字符: {raw[:200]}")
        return intent, latency, raw
    except Exception as e:
        latency = (time.time() - t0) * 1000
        error_msg = f"{type(e).__name__}: {e}"
        print(f"  [LLM DEBUG] ❌ API 异常 ({latency:.0f}ms): {error_msg}")
        return None, latency, error_msg


def generate_sensitivity_hint(net, violated_constraints, sensitivity):
    """
    生成灵敏度反馈提示
    Surgery #3: 只保留最关键的 2 条越限的灵敏度信息，避免信息过载
    """
    hints = []
    gen_indices = sensitivity['gen_indices']
    count = 0
    
    # 只处理前 6 个最严重的越限
    for viol in violated_constraints[:6]:
        if not isinstance(viol, tuple) or len(viol) < 2:
            continue
        viol_type, idx = viol[0], viol[1]
        if viol_type == 'voltage':
            sens = sensitivity['dV_dVset'][idx, :]
            top_gens = np.argsort(-np.abs(sens))[:2]
            gen_names = [f"Gen {gen_indices[g]} (Bus {net.gen.at[gen_indices[g], 'bus']}, Sensitivity={sens[g]:.3f})" for g in top_gens]
            hints.append(f"Bus {idx} Voltage → Preferred: {', '.join(gen_names)}")
            count += 1
        elif viol_type == 'thermal':
            sens = sensitivity['dLoading_dPg'][idx, :]
            top_gens = np.argsort(-np.abs(sens))[:2]
            gen_names = [f"Gen {gen_indices[g]} (Sensitivity={sens[g]:.2f}%/MW)" for g in top_gens]
            hints.append(f"Line {idx} Overload → Preferred: {', '.join(gen_names)}")
            count += 1
            
        if count >= 6:
            break
    
    return "\n".join(hints) if hints else None
