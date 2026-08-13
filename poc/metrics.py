"""
Unified metrics for evaluating power system control actions.
Ensures consistency between different solvers (LLM, OPF, Math).
"""

def evaluate_action_cost(actions, lambda_v=100.0, lambda_p=1.0, lambda_n=0.0):
    """
    Standardized evaluation metric (Evaluation Metric != Solver Objective).
    
    Args:
        actions (dict): Dictionary of {gen_id: {'dp': mw, 'dv': pu}}
        lambda_v (float): Weight for voltage setpoint changes. 
                         Standard: 100.0 (0.01 p.u. ~ 1 MW).
        lambda_p (float): Weight for active power redispatch.
        lambda_n (float): Penalty for number of actuators used (sparsity penalty).
        
    Returns:
        dict: Breakdown of costs.
    """
    if not actions:
        return {
            "cost_dp": 0.0,
            "cost_dv": 0.0,
            "n_act": 0,
            "cost_total": 0.0,
        }
        
    cost_dp = sum(abs(a.get('dp', 0.0)) for a in actions.values())
    cost_dv = sum(abs(a.get('dv', 0.0)) for a in actions.values())
    
    # Actuator count (ignoring trivial noise)
    n_act = sum(
        1 for a in actions.values()
        if abs(a.get('dp', 0.0)) > 1e-4 or abs(a.get('dv', 0.0)) > 1e-5
    )
    
    cost_total = lambda_p * cost_dp + lambda_v * cost_dv + lambda_n * n_act
    
    return {
        "cost_dp": float(cost_dp),
        "cost_dv": float(cost_dv),
        "n_act": int(n_act),
        "cost_total": float(cost_total),
    }
