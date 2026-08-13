# LLM-RA：基于语义选机与 SLP 物理映射的紧急 remedial action（论文工作说明）

IEEE 39-bus 系统上的 Proof-of-Concept。本文档描述**完整方法流程、代码结构与当前实验结果**（安全校验仅含 **Check-0 / Check-1**，不含 Check-2）。

---

## 1. 研究目标

在 **N-1 故障 + 负荷扰动** 后的越限场景下，快速给出 **发电机调节动作**（ΔP、ΔVset），使全网在稳态 AC 潮流下满足电压与热约束，并尽量降低控制成本。

**核心思路**：LLM 只做**语义层选机**（选哪几台发电机参与调节）；**数值求解**由灵敏度 + SLP 完成；**物理安全**由 Check-0/1 把关；失败则 **Fallback** 切负荷兜底。

---

## 2. 端到端流程

```
场景生成 (scenario.py)
    ↓  net：故障后 IEEE39 + 潮流收敛
越限检测 get_violations(net)
    ↓  无越限 → 结束
灵敏度 compute_sensitivity(net)          ← 非 OPF 模式
    ↓
意图生成：LLM (intent_engine) 或 规则 (B6)
    ↓  intent['generators'] + targets（targets 仅用于 LLM 输出合法性）
SLP 映射 intent_to_action                ← 最多 10 轮内循环
    ↓  actions: {gen_id: {dp, dv}}
Check-0 容量边界
    ↓  失败 → 带 hint 重试（LLM 最多 5 次 pipeline 轮；Rule/OPF 不重试）
Check-1 施加动作后 ACPF + 全约束再检
    ↓  失败 → 带 hint 重试（同上）
成功 → evaluate_action_cost 记录 J_eval
失败耗尽 → apply_fallback（切负荷 + 电压向 1.0 收缩）
```

### 2.1 场景生成（`poc/scenario.py`）

| 输入 | 输出 |
|------|------|
| `n_scenarios`, `seed` | `scenarios`（每项含 `net`, `lines_removed`, `load_scale_mean`, `category`, `info`） |

- 基准：`pandapower` **IEEE 39-bus**（`case39`）。
- **扰动**：单线 N-1 + 负荷约 1.05–1.30×。
- **保留**：过载 105–125%，越限数 ≤ 6，ACPF 收敛且存在越限。
- **丢弃**：不收敛、无越限、过载或越限数不在上述区间。

### 2.2 越限检测（`get_violations`）

| 输入 | 输出 |
|------|------|
| 已潮流的 `net` | `voltage_low` / `voltage_high` / `thermal` / `total_count` |

阈值：电压 **0.94–1.06 p.u.**；N-1 紧急热限 **110%**。

### 2.3 灵敏度（`compute_sensitivity`）

在**当前潮流点**数值微分，得到：

- `dV_dVset`：母线电压对发电机 **vm_pu** 的灵敏度  
- `dLoading_dPg`：线路负载率对发电机 **p_mw** 的灵敏度  

用途：规则/LLM **选机**；SLP **拼雅可比**。

### 2.4 意图层（`poc/intent_engine.py`）

**LLM 输入**：越限摘要、各机 P/V 与可调范围、按违限排序的灵敏度表、可选 retry `hint`。

**LLM 输出（JSON）**：
```json
{"generators": [0, 1], "targets": [...], "action_type": "auto"}
```

`generators` 为 pandapower **`net.gen` 行号**（IEEE 39 的 `case39` 为 **0–8**，对应母线 29/31–38；平衡机是 `ext_grid`，不在 `gen` 表里）。

**进入 SLP 的实质信息**：仅 **`generators` 列表**。`targets` 用于约束 LLM 输出格式，**不参与** `intent_to_action` 的数值映射。

**规则基线 B6**：按违限严重度 + 灵敏度 top-k 选机，最多 5 台。选机确定性，Check 失败后不重试，直接 Fallback。

### 2.5 SLP 物理映射（`intent_to_action`）

在 `intent['generators']` 子空间内，最多 **10 轮**内循环：

每轮顺序：
1. **灵敏度**（第 0 轮可用 pipeline 预计算；之后 `compute_sensitivity(work_net)`）
2. **`get_violations(work_net)`** — 若为 0 则成功退出
3. 电压越限 → 用 `dV_dVset` 更新 **ΔVset**（低压升、高压降）；线路热过载 → 用 `dLoading_dPg` 解 `J ΔP ≈ −excess` 更新 **ΔP**
4. **`cumulative` 累加**（相对场景初始 `net`，不每轮 reset）
5. 末尾 **ACPF**：`net + cumulative` → `runpp`；不收敛则 **回退本轮 cumulative 并 break**

| 输入 | 输出 |
|------|------|
| `net`, `intent`, `sensitivity` | `actions`: `{gen: {dp, dv}}` |

### 2.6 Check-0 / Check-1（`poc/physics_gateway.py`）

| 校验 | 作用 | 失败处理 |
|------|------|----------|
| **Check-0** | 动作后 P ∈ [Pmin, Pmax]、Vset ∈ **[0.94, 1.10]** | LLM：pipeline 重试 + hint；Rule/OPF：Fallback |
| **Check-1** | 将 actions 施加到 `net` 后 **ACPF**，再 `get_violations` 须为 0 | 同上 |

SLP **内循环不含** Check-0/1；二者在 SLP 返回后执行。Check-0 电压盒与 SLP clip / LLM prompt 一致。

### 2.7 Fallback（`apply_fallback`）

迭代：发电机 Vset 向 1.0 收缩 + 每轮切 5% 负荷（总切负荷上限 **20%**），最多 20 轮。

### 2.8 评估指标（`poc/metrics.py`）

与求解器内部目标解耦的统一外部成本：

**J_eval = Σ|ΔP| + 100·Σ|ΔV|**

另统计调节机组数 **n_act**。

---

## 3. 代码结构

在仓库根目录执行命令（相对路径，clone 后即可用）：

```
LLM_N1_Grid_Control/
├── README.md                # 英文项目说明（GitHub 首页）
├── docs/
│   └── paper.md             # 本文档（中文说明书）
├── poc/                     # 可运行 Python 包
│   ├── scenario.py          # 场景生成
│   ├── physics_gateway.py   # 越限、灵敏度、SLP、Check-0/1、Fallback
│   ├── intent_engine.py     # LLM + 规则意图
│   ├── pipeline.py          # Algorithm 1 主循环
│   ├── metrics.py           # J_eval
│   ├── math_solver.py       # OPF / 数学基线求解器
│   └── run_poc.py           # 多基线对比实验入口
├── tests/
│   └── test_retry_semantics.py
└── requirements.txt
```

**运行实验**（无 API 会跳过 LLM-RA，仍跑 OPF / Rule+SLP）：

```bash
cd LLM_N1_Grid_Control
python -m poc.run_poc
# LLM-RA 另需: export ANTHROPIC_API_KEY='sk-ant-...'
```

---

## 4. 实验设置

- **系统**：IEEE 39-bus New England  
- **场景**：**20** 组 N-1 + 负荷扰动（`seed=42`）  
- **时限**：单场景 `t_max=30s`；LLM pipeline 最多 **5** 次意图重试  

### 4.1 对比方法

| 模式 | 说明 |
|------|------|
| **AC-OPF (L2)** | 事后校正 OPF，内部 L2 目标 |
| **AC-OPF (L1)** | 近 L1 稀疏目标，公平对标 J_eval |
| **ΔV-only OPF** | 锁死有功，仅调电压设定（对标纯电压控制范式） |
| **Rule+SLP (B6)** | 规则选机 + 同一套 SLP（LLM 消融） |
| **LLM-RA（本文）** | Claude 选机 + SLP + Check-0/1 |

---

## 5. 当前实验结果（无 API 重跑，`seed=42`，20 场景）

LLM-RA 未重跑（无 API key）。下列为与当前代码一致的 **AC-OPF / Rule+SLP** 结果（统一 `evaluate_action_cost`）。

### 5.1 全场景（20 组）

| 指标 | AC-OPF (L2) | AC-OPF (L1) | ΔV-only OPF | Rule+SLP |
|------|-------------|-------------|-------------|----------|
| 意图成功（无 Fallback） | 17 (85%) | 15 (75%) | 8 (40%) | 15 (75%) |
| Fallback 成功 | 3 (15%) | 5 (25%) | 12 (60%) | 4 (20%) |
| Fallback 失败 | 0 | 0 | 0 | 1 (5%) |
| **整体成功率** | **100%** | **100%** | **100%** | **95%** |
| 平均延迟 | 306 ms | 321 ms | 339 ms | 1410 ms |
| 意图成功时平均 Σ\|ΔP\| | 57.2 MW | 49.0 MW | **0.0 MW** | 2.1 MW |
| 意图成功时平均 Σ\|ΔV\| | 0.1098 p.u. | 0.1077 p.u. | 0.1043 p.u. | **0.0370 p.u.** |
| 意图成功时平均 **J_eval** | 68.2 | 59.8 | 10.4 | **5.8** |
| 意图成功时平均 n_act | 9.0 | 9.0 | 9.0 | **3.2** |

### 5.2 对照解读

1. **L2-OPF** 意图成功率最高，但 J_eval 高：9 台机都动，有功再分配大。  
2. **L1-OPF** 略降 J_eval，意图成功率也略降。  
3. **ΔV-only OPF** 锁死 ΔP 后意图成功率掉到 40%（热过载场景无法只靠电压清零），成功时 Σ\|ΔP\|=0。  
4. **Rule+SLP** 与 LLM 同套物理映射：稀疏、低 J_eval；意图成功率 75%，整体 95%（场景 8 Fallback 失败）。  
5. 旧版 LLM-RA 数字（意图 14/20、J_eval≈2.6）**不能**当作当前代码结果；充值 API 后需重跑。

### 5.3 主要结论

1. 低成本主要来自 **稀疏电压控制**，Rule+SLP 已能体现，不依赖 LLM 全局优化。  
2. Check-0/1 后 OPF 系列不重试，失败即 Fallback；Rule 同样不重试。  
3. 内部 OPF 目标（L2 / 近 L1）与外部 J_eval 不一致，故并列 L1-OPF、ΔV-only 作对照。

---

## 6. 论文可强调的贡献点

1. **分层架构**：语义选机（LLM/规则）+ 物理 SLP 映射 + Check-0/1 校验 + Fallback。  
2. **控制范式**：少量机组、以 **ΔVset** 为主的紧急校正，在 J_eval 下成本极低。  
3. **可复现场景集**：N-1 + 负荷扰动生成器（过载/越限筛选）。  
4. **多基线对比**：OPF(L2/L1)、纯电压 OPF、Rule+SLP 消融。

---

## 7. 已知局限

- LLM 延迟高（秒级），不适合硬实时。  
- SLP 为局部线性化，依赖选机质量；潮流不收敛则回退本轮并提前终止。  
- 变压器热过载在 SLP 中无 `dLoading` 灵敏度行（仅线路）。  
- 实验为单 seed、20 场景，需扩展多 seed 统计检验。

---

*最后更新：与仓库 `poc/` 代码一致；已移除 OOD；Check-2/GNN 动态前瞻已从本仓库移除。*
