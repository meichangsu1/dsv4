如果现在让我重新设计一个异步 RL 系统，我不会设计成单纯的：

```text
Rollout → Queue → Trainer
```

也不会把所有逻辑都塞进一个 Dataflow 大服务里。我会设计成：

> **Thin Controller + Medium Dataflow + Service Roles + Versioned TransferQueue**

也就是：**Controller 管服务生命周期，Dataflow 管数据语义，TransferQueue 管高速数据流，Trainer 只管优化，Rollout 只管生成，WeightSync 只管权重版本。**

---

## 1. 设计目标

异步 RL 系统首先要解决这几个问题：

```text
1. rollout 长尾导致 trainer idle
2. rollout / train / reward / advantage / ref logprob 资源不匹配
3. 异步后样本 stale，影响 PPO/GRPO 效果
4. 多轮 agent / tool / env 带来不可预测延迟
5. 多策略、多 env、多 backend 时 trainer loop 过重
6. 训练、推理、权重同步、数据流需要可观测和可恢复
```

AReaL 的 fully async 证明了 rollout 和 training 完全解耦可以显著提升吞吐：rollout workers 持续生成，training workers 收够 batch 就更新，并用 workload balance 和 staleness-enhanced PPO 控制 stale data；论文报告最高 2.57× speedup。([arXiv][1])

veRL fully async 也采用 Rollouter、MessageQueue、Trainer、ParameterSynchronizer 的分离架构，支持 sample-level streaming、staleness_threshold、partial rollout 和 rollout-side old logprob；文档报告 Qwen2.5-7B 128 GPU 上 2.35×–2.67× 提升。([Verl][2])

Relax 则给了另一个方向：Controller + Ray Serve service roles + TransferQueue + DCS weight sync，把 Actor、Rollout、ActorFwd、Reference、Advantages 都服务化，通过 TransferQueue 流式交换数据。([GitHub][3])

AstraFlow 的启发是：不要只做计算解耦，还要把 rollout scheduling、data selection、routing、replay、staleness correction 从 trainer loop 里拿出来，变成 Dataflow Layer 的职责。([arXiv][4])

---

## 2. 总体架构

我会这样设计：

```text
                         ┌──────────────────────────┐
                         │      Thin Controller      │
                         │ lifecycle / autoscale     │
                         │ fault recovery / config   │
                         │ checkpoint orchestration  │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │          Dataflow Plane            │
                    │ metadata / routing / sampling      │
                    │ staleness / replay / filtering     │
                    │ batch policy / multi-policy stream │
                    └────────────┬────────────┬─────────┘
                                 │            │
              rollout tasks      │            │ train batches
                                 │            │
                    ┌────────────▼───┐    ┌───▼────────────┐
                    │ Rollout Service│    │ Trainer Service│
                    │ SGLang/vLLM    │    │ Megatron/FSDP  │
                    │ AgentLoop/Env  │    │ PPO/GRPO/...   │
                    └────────────┬───┘    └───┬────────────┘
                                 │            │
                                 │ trajectory │ weights/version
                                 ▼            ▼
                    ┌────────────────────────────────────┐
                    │          TransferQueue              │
                    │ hot streaming data path             │
                    │ samples / logprobs / rewards        │
                    └────────────────┬───────────────────┘
                                     │
                    ┌────────────────▼───────────────────┐
                    │      Trajectory Store + Metadata DB │
                    │ cold data / replay / audit / debug  │
                    └────────────────┬───────────────────┘
                                     │
                    ┌────────────────▼───────────────────┐
                    │          Weight Manager             │
                    │ version / delta sync / DCS/NCCL     │
                    └────────────────────────────────────┘
```

核心原则：

```text
Controller 不决定训练数据分布
Dataflow 不训练模型
TransferQueue 不做复杂算法决策
Trainer 不调度 rollout
Rollout 不知道 batch 怎么采样
Weight Manager 不管数据选择
```

---

## 3. 组件职责

### 3.1 Thin Controller：只管服务，不管数据策略

Controller 负责：

```text
启动 / 停止服务
扩缩容 rollout pool
健康检查
故障恢复
checkpoint orchestration
配置下发
全局状态监控
```

它不应该负责：

```text
哪些样本训练
哪些样本 replay
staleness 怎么降权
多策略数据怎么路由
batch 里 fresh/stale/long-tail 比例
```

Relax 的 Controller 做 training loop 和 global restart，Service 管 placement groups/lifecycle，Registry 管 role 和 algorithm mapping，这个工程边界是有价值的。([GitHub][3])

---

### 3.2 Medium Dataflow：管数据语义，不做大泥球

Dataflow Plane 是这个系统的“数据控制面”。它负责：

```text
prompt/task pool
rollout task routing
trajectory metadata
policy_version / staleness
fresh buffer
replay buffer
filter policy
batch sampler
multi-policy routing
long-tail bucket
backpressure
```

但它不负责：

```text
PPO/GRPO loss
env.step 具体执行
reward model forward
权重实际传输
GPU placement
```

我会把 Dataflow 做成 **medium**，不是 heavy。它的核心接口是：

```python
class DataflowPlane:
    def pull_rollout_tasks(self, worker_state) -> list[RolloutTask]:
        ...

    def push_trajectories(self, trajectories: list[Trajectory]) -> None:
        ...

    def pull_train_batch(self, policy_id: str, trainer_state, batch_spec) -> TrainBatch:
        ...

    def ack_train_result(self, batch_ids, metrics) -> None:
        ...

    def update_policy_version(self, policy_id: str, version: int) -> None:
        ...
```

veRL 的 TransferQueue 文档提到，TransferQueue 提供细粒度数据管理、load balancing、data gateway，并能把数据依赖从单 controller 中解耦出来；同时支持自定义 Sampler 和 StreamingDataLoader。这个方向很适合用作 Dataflow 的底层 hot path。([Verl][5])

---

### 3.3 TransferQueue：高速热数据通路

TransferQueue 只做高吞吐数据交换：

```text
Rollout 写入：
  tokens
  old_logprobs
  rewards
  masks
  metadata

Trainer 消费：
  streaming mini-batch
  rank-aware shard
  micro-batch
```

我会把 TransferQueue 设计为：

```text
Hot path:
  最近生成、马上训练的数据

Not source of truth:
  长期存储、replay、debug 走 Trajectory Store

Metadata visible:
  policy_version
  task_type
  length
  reward
  staleness
  status
```

TransferQueue 不能只是 FIFO，因为异步 RL 需要按 freshness、policy_id、task_type、reward、长度、long-tail 等条件采样。veRL TransferQueue 文档中提到它用 TransferQueueController 跟踪样本生产/消费状态，支持用户自定义 Sampler，并提供 StreamingDataLoader 让各 rank 无需 single-controller intervention 自动消费数据。([Verl][5])

---

### 3.4 Rollout Service：只管生成和环境交互

Rollout Service 负责：

```text
从 Dataflow 拉 task
加载指定 policy version
执行 generate / agent loop
调用 Env / Tool / Reward
计算 rollout old_logprobs
把 trajectory 写入 TransferQueue / Store
```

接口：

```python
class RolloutService:
    def refresh_weights(self, policy_id: str, version: int):
        ...

    def run_task(self, task: RolloutTask) -> Trajectory:
        ...

    def report_status(self) -> WorkerState:
        ...
```

对于多轮 agent：

```python
class BaseInteractionEnv:
    def reset(self, task) -> Observation:
        ...

    def step(self, action) -> tuple[Observation, Reward, Done, Info]:
        ...

    def format_observation(self, obs) -> ModelInput:
        ...
```

Relax 的 Agentic RL 支持 multi-turn sampling、loss masking、BaseInteractionEnv，以及把 model outputs 和 environment observations 区分开，只有模型动作参与训练；这个设计非常值得吸收。([GitHub][3])

---

### 3.5 Trainer Service：只管优化

Trainer 不调度 rollout，不决定 prompt，不直接控制 env。

它只做：

```text
从 Dataflow/TransferQueue 拉 batch
recompute current logprobs
计算 advantages
计算 PPO/GRPO/GSPO/SAPO loss
更新模型
发布新 weight version
上报 metrics
```

接口：

```python
class TrainerService:
    def pull_batch(self) -> TrainBatch:
        ...

    def train_step(self, batch: TrainBatch) -> TrainResult:
        ...

    def publish_weights(self, result: TrainResult) -> WeightVersion:
        ...
```

---

### 3.6 Weight Manager / ParameterSynchronizer：只管权重流

权重同步和数据队列要分开。

数据流：

```text
Rollout → TransferQueue → Trainer
```

权重流：

```text
Trainer → Weight Manager / ParameterSynchronizer → Rollout / Ref / ActorFwd
```

veRL fully async 里 ParameterSynchronizer 是 Trainer 和 Rollouter 之间做 NCCL 参数同步的组件，而 MessageQueue 只是暂存 Rollouter 生成的 samples；Trainer 拉够样本训练若干轮后触发参数同步。([Verl][2])

所以我会明确画成：

```text
数据路径：
Rollout ─────▶ TransferQueue ─────▶ Trainer

权重路径：
Trainer ─────▶ WeightManager ─────▶ Rollout / Ref / ActorFwd
```

---

## 4. 核心运行流程

### 4.1 正常异步训练流程

```text
1. Controller 启动 Rollout / Trainer / Env / Reward / WeightManager
2. Dataflow 从 prompt pool 选择 task
3. Rollout 拉 task，用 policy_vK 生成 trajectory
4. Rollout 写入 old_logprobs、reward、mask、policy_version
5. Dataflow 更新 metadata
6. Trainer 通过 StreamingDataLoader 拉 batch
7. Dataflow sampler 保证 fresh/stale/long-tail 比例
8. Trainer 用 current policy 计算 new_logprobs
9. PPO/GRPO 用 rollout old_logprobs 做 ratio
10. Trainer 更新模型，发布 policy_vK+1
11. WeightManager 异步同步给 Rollout / Ref / ActorFwd
12. Dataflow 根据版本更新 staleness
```

---

## 5. 最关键的策略：bounded async

我不会默认 fully unrestricted async，而是默认 **bounded async**。

```yaml
async:
  mode: bounded_async
  max_staleness: 1
  target_staleness_p95: 1
  stale_sample_policy: decay_or_drop
  partial_rollout: false
  refresh_weight_on_new_episode: true
```

原因是：异步 RL 的吞吐收益来自并行，但训练效果风险来自 stale data。AReaL 用 workload balance 和 staleness-enhanced PPO 来稳定旧样本；veRL 用 `staleness_threshold` 控制 stale samples 比例，并建议不要过大以免影响训练精度。([arXiv][1])

我的 staleness policy 会这样：

```python
def staleness_policy(sample, current_version):
    s = current_version - sample.rollout_policy_version

    if s <= 1:
        return Train(weight=1.0)
    elif s == 2:
        return Train(weight=0.5)
    else:
        return Drop(reason="too_stale")
```

生产上可以配置：

```yaml
staleness:
  max_staleness: 1
  mild_stale_max: 2
  mild_stale_ratio: 0.15
  drop_too_stale: true
```

---

## 6. 长尾怎么处理

长尾不能简单丢，否则训练分布会偏向短样本。

我会用四层机制：

### 6.1 Streaming：完成一条写一条

```text
短 trajectory 先进入 queue
trainer 凑够 batch 就训
长 trajectory 不阻塞整个 step
```

### 6.2 Long-tail bucket：长样本单独管理

```text
normal buffer
long-tail buffer
replay buffer
```

batch 里显式保留一部分长样本：

```yaml
dataflow:
  fresh_ratio: 0.75
  mild_stale_ratio: 0.15
  long_tail_ratio: 0.10
```

### 6.3 Trajectory-level consistency：默认一条轨迹不换模型

默认：

```text
一条 trajectory 用同一个 policy_version 跑完
```

这样 PPO/GRPO 的 old_logprob 语义最干净。

### 6.4 Partial rollout：可选，不默认

veRL fully async 的 partial rollout 会在参数同步时中断正在生成的 rollout，保存后续恢复，减少等待 active tasks 完成的时间。这个能进一步减少长尾等待，但会增加 trajectory 版本一致性的复杂度，所以我会作为高级开关。([Verl][2])

---

## 7. 数据结构

每条 trajectory 必须版本化：

```python
@dataclass
class Trajectory:
    trajectory_id: str
    prompt_id: str
    task_type: str
    env_id: str

    policy_id: str
    rollout_policy_version: int
    weight_hash: str

    input_ids: Tensor
    response_ids: Tensor
    attention_mask: Tensor
    loss_mask: Tensor

    old_logprobs: Tensor
    ref_logprobs: Optional[Tensor]

    reward: float
    reward_version: str
    advantage: Optional[Tensor]

    token_len: int
    turn_count: int
    created_at: float
    finished_at: float

    status: Literal[
        "completed",
        "trainable",
        "consumed",
        "replayable",
        "stale",
        "dropped",
        "failed",
    ]
```

最重要的是：

```text
old_logprobs 必须来自 rollout policy version
不能用当前 trainer policy 重算 old_logprob 冒充 old
```

veRL fully async 文档也强调，PPO/GRPO/DAPO 的 old_log_prob 与参数版本和 tokens 有隐含关联；fully async 默认 old_log_prob 由 rollout 计算，而不是 trainer 计算。([Verl][2])

---

## 8. Dataflow Sampler

Dataflow sampler 是系统效果的关键。

我会让 Trainer 请求的是：

```python
BatchSpec(
    policy_id="actor",
    batch_size=1024,
    max_staleness=1,
    fresh_ratio=0.8,
    long_tail_ratio=0.1,
    task_mix={"math": 0.5, "code": 0.3, "search": 0.2},
)
```

Dataflow 返回已经组织好的 batch：

```python
def sample_train_batch(spec, current_version):
    fresh = index.query(
        policy_id=spec.policy_id,
        staleness_lte=1,
        status="trainable",
    )

    mild_stale = index.query(
        policy_id=spec.policy_id,
        staleness_eq=2,
        status="trainable",
    )

    long_tail = index.query(
        policy_id=spec.policy_id,
        is_long_tail=True,
        staleness_lte=2,
    )

    batch = mix(
        fresh, ratio=0.75,
        mild_stale, ratio=0.15,
        long_tail, ratio=0.10,
    )

    return pack(batch)
```

这样 Trainer 不再做数据策略，只做训练。

---

## 9. 多角色扩展

我会支持这些 role：

```text
Rollout
Actor Trainer
Critic Trainer
Reference
ActorFwd / Logprob
Reward / GenRM
Advantage
Env / Tool
Verifier
Selector
TestGen
```

Relax 把 Actor、Rollout、Critic、ActorFwd、Advantages、GenRM 都作为 Ray Serve deployments，这种 role service 化很适合生产系统。([GitHub][3])

但我不会强制所有任务一开始都拆这么细。默认分两档：

```text
轻量模式：
  Rollout + Trainer + WeightManager + Dataflow

完整模式：
  Rollout + Actor + Ref + ActorFwd + Advantage + Reward + Env + Dataflow
```

---

## 10. 推荐默认配置

```yaml
system:
  mode: bounded_async
  controller: thin
  dataflow: medium
  transfer_queue: enabled
  trajectory_store: enabled
  metadata_db: enabled

rollout:
  backend: sglang_or_vllm
  max_inflight_per_worker: 64
  max_turns: 8
  max_tokens_per_episode: 8192
  soft_timeout_sec: 120
  hard_timeout_sec: 600
  refresh_weight_on_new_episode: true
  partial_rollout: false

trainer:
  backend: megatron_or_fsdp
  algorithm: grpo
  ppo_mini_batch_size: 1024
  train_when_batch_ready: true
  use_rollout_log_probs: true
  recompute_current_log_probs: true

async:
  max_staleness: 1
  mild_stale_max: 2
  mild_stale_ratio: 0.15
  stale_loss_decay: exp
  drop_too_stale: true
  trigger_parameter_sync_step: 1

dataflow:
  fresh_ratio: 0.75
  mild_stale_ratio: 0.15
  long_tail_ratio: 0.10
  replay_ratio: 0.00
  filter_zero_advantage: true
  filter_invalid_reward: true
  group_by_prompt: true

weight:
  sync_mode: delta_or_full_pull
  full_sync_interval: 20
  nccl_sync: true
  overlap_with_training: true

monitor:
  - trainer_idle_ratio
  - rollout_idle_ratio
  - queue_depth
  - queue_age_p95
  - staleness_p50
  - staleness_p95
  - stale_drop_rate
  - long_tail_ratio
  - kl
  - clip_ratio
  - reward_mean
  - eval_score
```

---

## 11. 为什么不是直接照搬某一个框架

我会吸收各家的优点：

| 来源        | 我会吸收                                                                       | 我会避免                               |
| --------- | -------------------------------------------------------------------------- | ---------------------------------- |
| AReaL     | fully async、staleness-aware training                                       | 只围绕 learner，不够 dataflow 化          |
| veRL      | 渐进异步、staleness_threshold、partial rollout、rollout old_logprob、TransferQueue | Trainer / single-controller 数据路径过重 |
| SLIME     | AsyncRolloutWorker 保持 in-flight pool，简单有效                                  | 异步主要局限在 rollout 边界                 |
| Relax     | Controller + service roles + TransferQueue + DCS，生产化好                      | Controller 不应塞太多数据策略               |
| AstraFlow | Dataflow Layer 做数据控制面                                                      | Dataflow 不能变成新的大泥球                 |

SLIME 的 fully-async rollout 设计很轻：后台 asyncio worker 保持固定 in-flight generations，下一步训练不用等最慢 in-flight sample；它适合作为 Rollout Service 内部实现。([thudm.github.io][6])

veRL TransferQueue 的动机是缓解 single controller `RayPPOTrainer` 的数据传输瓶颈，因为所有 `DataProto` 都经由 RayPPOTrainer 会形成单点瓶颈；这说明新系统从一开始就应该把数据流从 trainer/controller 中拆出来。([Verl][5])

Relax 的 TransferQueue + StreamingDataLoader + DCS weight sync 说明生产系统里数据流、权重流、服务编排应该分开；Actor 可以在 Rollout 增量写入 TransferQueue 时就开始消费样本，减少阶段间 GPU idle。([GitHub][3])

---

## 12. 最终架构一句话

我会设计成：

```text
Thin Controller
  管服务生命周期、扩缩容、恢复

Medium Dataflow
  管数据选择、路由、staleness、replay、filter、batch policy

TransferQueue
  管高速 streaming data path

Trajectory Store
  管可恢复、可 replay、可审计的数据

Trainer Service
  只管优化

Rollout / Env / Reward Service
  只管生成、交互和打分

Weight Manager
  只管版本和权重同步
```

最核心的判断标准是：

```text
Trainer 不应该拥有全局 RL pipeline。
Controller 不应该拥有训练数据分布。
Dataflow 不应该拥有训练算法。
TransferQueue 不应该拥有复杂策略。
```

这样系统既能解决长尾，又能控制 staleness，还不会把复杂度全部堆到一个组件里。

[1]: https://arxiv.org/abs/2505.24298?utm_source=chatgpt.com "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"
[2]: https://verl.readthedocs.io/en/latest/advance/fully_async.html "Recipe: Fully Async Policy Trainer — verl  documentation"
[3]: https://github.com/redai-infra/Relax "GitHub - redai-infra/Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale · GitHub"
[4]: https://arxiv.org/abs/2605.15565?utm_source=chatgpt.com "AstraFlow: Dataflow-Oriented Reinforcement Learning for Agentic LLMs"
[5]: https://verl.readthedocs.io/en/latest/data/transfer_queue.html "TransferQueue Data System — verl  documentation"
[6]: https://thudm.github.io/slime/_examples_synced/fully_async/README.html "Fully-Async Rollout Example — slime"
