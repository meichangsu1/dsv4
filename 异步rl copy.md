
## 1\. 异步 RL 主要解决什么问题

LLM RL 的瓶颈通常不是单纯的反向训练，而是 **rollout / generation / env interaction 太慢且长尾严重**。同步 PPO/GRPO/DAPO 的典型流程是：

```text {"data-theme":"githubLight"}
rollout 生成完整 batch
    ↓
reward / env / verifier
    ↓
trainer 更新
    ↓
同步权重
    ↓
下一轮 rollout
```

问题是 一个 batch 里不同样本生成时长差异极大：短样本几秒结束，长推理、多轮工具调用、代码执行、搜索任务可能几十秒甚至几分钟。同步训练必须等最慢样本完成，所以 trainer GPU 空等，rollout GPU 也会在阶段切换时空等。

因此异步 RL 的核心目标是：

```text {"data-theme":"githubLight"}
系统上：
  rollout 和 training overlap，长尾样本不阻塞 trainer

算法上：
  控制 stale data，尽量保持 PPO/GRPO 的训练效果

工程上：
  rollout、trainer、reward、env、weight sync 能独立扩缩容和替换
```

但异步会带来一个新问题：**off-policy / staleness**。样本生成时用的是旧 policy，训练时模型可能已经更新了多轮。所以成熟系统都会在吞吐和训练效果之间做权衡：用 `staleness_threshold`、`max_staleness`、old logprob、importance correction、partial rollout、dataflow policy 等机制控制偏差。

---

## 2\. 总览对比
| 框架 | 要解决的主要问题 | 核心方案 | Global orchestrator | 数据通路 | 训练效果/系统效果 |
|--|---------|---------|-----|--------|------------------------|
| **AReaL** | 同步 RL 中 generation 长尾导致 trainer idle | fully async：rollout workers 持续生成，training workers 收够 batch 就更新 | learner / training-side async coordinator | rollout → experience buffer/queue → trainer | 论文报告最高 **2\.57× training speedup**，最终效果匹配或更好；README 还提到 v0.3 版本 **2\.77× speedup** ([arXiv](https://arxiv.org/abs/2505.24298?utm_source=chatgpt.com "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning")) |
| **veRL async** | 从同步 PPO/GRPO 平滑迁移到异步，降低长尾 bubble | one-step-off + fully async；Rollouter、MessageQueue、Trainer、ParameterSynchronizer | `RayPPOTrainer` / `FullyAsyncTrainer` | Rollouter → MessageQueue/TransferQueue → Trainer | fully async 文档报告 Qwen2.5-7B 128 GPU 上 **2\.35×–2.67×** 提升，且结果影响不显著 ([Verl](https://verl.readthedocs.io/en/latest/advance/fully_async.html "Recipe: Fully Async Policy Trainer — verl  documentation")) |
| **SLIME** | SGLang/Megatron 栈中 rollout 长尾、agentic 生成慢 | fully-async rollout worker 保持固定 in-flight generation | `train_async.py` + `AsyncRolloutWorker` | data_buffer / output queue → training driver | 下一轮训练不等最慢 in-flight sample；`ABORTED` 样本回到 data_buffer ([Thudm](https://thudm.github.io/slime/\_examples_synced/fully_async/README.html "Fully-Async Rollout Example — slime")) |
| **AstraFlow** | 不是只解决长尾，而是解决 trainer-centered 导致的多策略/弹性/异构/数据算法扩展困难 | Dataflow Layer + RaaS + Trainer + Weight Manager | **Dataflow Layer** | RaaS ↔ Dataflow ↔ Trainer，Weight Manager 异步权重流 | 多策略训练时间 **2\.7×** 加速；支持多策略、弹性、异构/跨区域、数据算法组合 |
| **Relax** | 生产化 omni-modal RL 中训练、推理、优势计算、ref 等角色耦合；多模态/多轮场景吞吐低 | Controller + Ray Serve services + TransferQueue + DCS weight sync | **Controller** | Rollout/Actor/ActorFwd/Reference/Advantages 通过 TransferQueue 流式交换 | TransferQueue 解耦训练和推理；`--max-staleness` 控制 off-policy；支持动态增删 rollout engines ([GitHub](https://github.com/redai-infra/Relax "GitHub - redai-infra/Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale · GitHub")) |



## 3\. AReaL：fully async，重点解决 generation-training 互相等待

### 解决什么问题

AReaL 主要针对同步 LLM RL 的系统低效：generation 必须等 batch 中最长输出完成，training 才能开始。对于 math/code reasoning，输出长度差异大，长尾严重，导致 trainer GPU 利用率低。([arXiv](https://arxiv.org/abs/2505.24298?utm_source=chatgpt.com "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"))

### 采用什么架构

AReaL 的核心是 **完全解耦 generation 和 training**：

```text {"data-theme":"githubLight"}
Rollout Workers:
  不停生成新样本，不等待 trainer

Training Workers:
  收集到 batch 后立即更新模型

Weight Sync:
  新权重异步回流给 rollout workers

Staleness Control:
  workload balancing 控制 stale data
  staleness-enhanced PPO 处理旧样本
```

所以它不是传统 trainer 调 rollout、等 rollout、再训练的同步 loop，而是 **learner-driven fully async**。它的 global orchestrator 更偏 training side / learner side：trainer 不再直接阻塞 rollout，但系统的主要节奏仍围绕 training worker 更新和权重回流展开。

### 最终效果

AReaL 论文报告，在 math/code reasoning benchmarks 上，相比最优同步系统，在同等 GPU 下最高 **2\.57× training speedup**，最终性能匹配或更好。GitHub README 还提到 v0.3 “boba²” 版本实现 **2\.77× speedup**，并支持 agentic RL、自定义 agent runtime、math/code/search/customer-service 等任务。([arXiv](https://arxiv.org/abs/2505.24298?utm_source=chatgpt.com "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"))

---

## 4\. veRL async：从 one-step-off 到 fully async，重点是渐进式异步和可控 staleness

### 解决什么问题

veRL async 的出发点也是同步 PPO/GRPO/DAPO 中 rollout 长尾导致 GPU idle。它的优势是路线比较稳：先 one-step-off，再 fully async，不要求一开始就完全异步。veRL 文档明确说 separated rollout/train architecture 能更灵活分配资源，缓解 long-tail 导致的低 GPU 利用率；one-step-off 能缓解长 rollout 时间，但不能完全消除长尾影响，所以 fully_async_policy 进一步支持 streaming 和 partial rollout。([Verl](https://verl.readthedocs.io/en/latest/advance/fully_async.html "Recipe: Fully Async Policy Trainer — verl  documentation"))

### 采用什么架构

veRL fully async 由四个核心组件组成：

```text {"data-theme":"githubLight"}
Rollouter
  sample-by-sample 生成数据

MessageQueue
  暂存 rollout 生成的样本

Trainer
  从 MessageQueue 拉样本
  凑够 require_batches * ppo_mini_batch_size 后训练

ParameterSynchronizer
  负责 Trainer 和 Rollouter 间的 NCCL 参数同步
```

Rollouter 逐样本生成并放进 MessageQueue；Trainer 逐样本拉取，达到指定 batch 规模后训练；训练若干轮后触发 ParameterSynchronizer 同步参数。([Verl](https://verl.readthedocs.io/en/latest/advance/fully_async.html "Recipe: Fully Async Policy Trainer — verl  documentation"))

### 关键控制参数

veRL async 的参数体系很清楚：

```text {"data-theme":"githubLight"}
async_training.staleness_threshold
  控制允许 stale samples 的比例；0 表示同步，>0 表示异步

async_training.require_batches
  Trainer 每次拉多少个 mini-batch 后训练

async_training.trigger_parameter_sync_step
  Trainer 本地更新多少轮后同步参数

async_training.partial_rollout
  参数同步时是否中断/保存进行中的 rollout，减少等待 active tasks 的时间

actor_rollout_ref.actor.use_rollout_log_probs=True
  使用 rollout 时对应参数版本生成的 old_logprob，保证 PPO/GRPO 重要性采样语义
```

old logprob 和参数版本、tokens 有隐含关联，fully async 中默认 old_log_prob 应由 rollout 侧计算，而不是 trainer 用当前参数重算。

### 最终效果

veRL fully async 文档报告：在 Qwen2.5-7B、128 GPU 训练中达到 **2\.35×–2.67×** 性能提升，且没有显著影响结果；另一个实验描述在 32/64/128 卡下约 **2×** 提升。([Verl](https://verl.readthedocs.io/en/latest/advance/fully_async.html "Recipe: Fully Async Policy Trainer — verl  documentation"))

---

## 5\. SLIME：fully-async rollout，重点是高性能 Megatron + SGLang 栈里的生成长尾

### 解决什么问题

SLIME 的 fully-async 示例重点解决的是 rollout 边界上的长尾：标准 pipeline 下一轮训练可能要等最慢的样本，而 fully-async rollout 用后台 worker 持续维护一批 in-flight generations，让下一轮训练不被最慢样本卡住。([Thudm](https://thudm.github.io/slime/\_examples_synced/fully_async/README.html "Fully-Async Rollout Example — slime"))

### 采用什么架构

SLIME fully async 的核心是：

```text {"data-theme":"githubLight"}
train_async.py
  异步训练 driver

generate_rollout_fully_async
  替换普通 rollout function

AsyncRolloutWorker
  进程级后台 worker
  thread + asyncio loop
  保持固定数量 in-flight generations
  完成 group 写入 output queue
  ABORTED 样本重新放回 data_buffer
```

SLIME 更像是 **高性能训练/生成底座 + async rollout 插件**，Relax基于此

---

## 6\. Relax：Controller + service-oriented + TransferQueue，重点是生产化、多模态、多轮、弹性 rollout

### 解决什么问题

Relax 面向的是更工程化的生产问题：omni-modal RL、多轮 agent、训练/推理/Ref/Advantage 多角色并行、服务弹性和容错。它强调训练和推理完全解耦，同时支持文本、图像、视频、音频的端到端 RL post-training。([GitHub](https://github.com/redai-infra/Relax "GitHub - redai-infra/Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale · GitHub"))

### 采用什么架构

Relax 不是 trainer-centered（trainer 不是协调者），而是：

```text {"data-theme":"githubLight"}
Controller
  training loop、global restart、服务编排

Service / Registry
  placement group、lifecycle、role & algorithm mapping

Ray Serve deployments
  Actor、Rollout、Critic、ActorFwd、Advantages、GenRM

TransferQueue
  Rollout、Actor、ActorFwd、Reference、Advantages 间流式交换数据

DCS
  Actor 训练后通过 NCCL/GLOO 给 Rollout/ActorFwd/Reference 同步权重
```

Fully Async 模式下，Actor、Rollout、ActorFwd、Reference、Advantages 在独立 GPU 集群上并行，通过 TransferQueue 交换数据，并通过 DCS 异步同步权重。([GitHub](https://github.com/redai-infra/Relax "GitHub - redai-infra/Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale · GitHub"))

### 关键机制

Relax 的几个核心机制是：

```text {"data-theme":"githubLight"}
TransferQueue
  Rollout 增量写入样本，Actor StreamingDataLoader 开始消费，减少 phase 间 GPU idle

--max-staleness
  控制 off-policy training data drift，在 on-policy accuracy 和 throughput 间折中

DCS weight sync
  每个 training step 后把权重 broadcast 到 Rollout / ActorFwd / Reference，并与下一步训练 overlap

BaseInteractionEnv
  reset / step / format_observation，把环境和 rollout 解耦

Elastic Rollout Scaling
  训练中通过 HTTP REST API 动态增加/减少 inference engines
```

Relax 文档明确说 `--max-staleness` 精确控制 off-policy 数据漂移；多轮 agentic RL 中用 loss masking 区分模型输出和环境 observation，只有模型动作参与训练；并支持动态增删 inference engines。([GitHub](https://github.com/redai-infra/Relax "GitHub - redai-infra/Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale · GitHub"))

### 最终效果

Relax README 没有在首页给出类似 AReaL/veRL 的统一 speedup 数字，但它实现的效果是系统能力层面的：

```text {"data-theme":"githubLight"}
训练和推理解耦
多角色并行
TransferQueue 流式数据交换
可配置 staleness
多轮 env 解耦
omni-modal RL
动态 rollout 扩缩容
生产级 health/restart/metrics
```

这些能力更偏生产化，而不是单个 benchmark 的 2.x 倍报告。([GitHub](https://github.com/redai-infra/Relax "GitHub - redai-infra/Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale · GitHub"))

---

## 7\. AstraFlow：从“解决长尾”升级到“解决 trainer-centered 的系统扩展问题”（**下一代 dataflow-oriented agentic RL 系统抽象**）

  提出目前的 LLM 强化学习框架可以分为**同机同步（Colocated Synchronous）**、**计算解耦异步（Disaggregated Async）**以及**面向数据流的完全解耦（Dataflow-Oriented）**三个阶段。

面向数据流的完全解耦（Dataflow-Oriented）特点：彻底分离了计算与控制责任。

* **控制权分散**：Rollout 服务（RaaS）、训练器和数据流层各自运行独立的控制循环，仅通过数据和权重接口交互。
* **原生支持复杂负载**：无需修改系统代码即可支持多策略训练（如 Solver-Verifier）、跨地域算力调度和基于智能体（Claude Code）的自动扩缩容。

### 解决什么问题

AstraFlow 不只是要解决 rollout 长尾。它认为已有系统即使做了 rollout-training disaggregation，也常常只是把计算分开，但 **rollout scheduling、data selection、replay、staleness handling、weight sync 仍然嵌在 trainer-centered control loop 中**。这样一旦要支持多策略协同、弹性 rollout、异构/跨区域 rollout、数据算法组合，就需要 ad-hoc patch 或大规模重构。

### 采用什么架构

AstraFlow 把系统拆成：

```text {"data-theme":"githubLight"}
Dataflow Layer
  管 prompts、trajectories、metadata、training batches
  管 sampling、filtering、routing、replay、staleness correction

RaaS, Rollout-as-a-Service
  只负责消费 task、生成 trajectory、刷新权重

Trainer
  只负责从 Dataflow 拉 batch、优化模型、发布权重

Weight Manager
  负责模型版本、异步刷新、full/sparse/delta 权重传输
```

AstraFlow 的核心是把 disaggregation 从“计算分离”推进到“控制职责分离”：rollout services、trainers、dataflow layer 各自运行自治 control loop，只通过最小数据和权重接口交互。

与其他主流框架在关键能力上的横向对比：
| 属性 | **AstraFlow** | **AReaL** | **SLIME** | **verl (最新版)** | **RLBoost** | **Dr.MAS (verl)** | **prime-rl** |
|-----------|:------:|:-----:|:-----:|:-------:|:-----:|:-------:|:-----:|
| **多策略协作训练** | **✓** | ✗ | ✗ | ✗ | ✗ | **✓** | ✗ |
| **完全异步训练** | **✓** | **✓** | **✓** | **✓** | ✗ | ✗ | **✓** |
| **解耦 Rollout-训练架构** | **✓** | **✓** | **✓** | **✓** | **✓** | ✗ | **✓** |
| **运行时弹性 Rollout 缩放** | **✓** | ✗ | ✗ | ✗ | **✓** | ✗ | p |
| **跨地域 / 异构 Rollout** | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ | p |
| **可替换训练器/Rollout 服务** | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ | p |
| **模块化数据算法接口** | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |



*注：✓ 为完全支持；✗ 为不支持；p 为部分支持。*

### 3\. 关键性能指标对比

在来源的实验评估中，AstraFlow 展示了相对于现有系统的优势：

* **训练速度**：在多策略数学协作训练任务中，AstraFlow 的迭代时间（77.65s）比基于 verl 的 Dr. MAS（212.64s）快了 **2\.7 倍**。
* **效率对齐**：在单策略任务中，**AstraFlow 的性能和效率能与专门优化的 AReaL 持平**，但提供了更强的灵活性。
* **带宽优化**：通过稀疏权重更新，AstraFlow 将跨地域传输的数据量压缩了 **30 倍以上**。
| 维度 | 传统 async RL / AReaL / veRL async | AstraFlow |
|--------|-------------------|------------------|
| 控制中心 | 通常还是 trainer 或 trainer-side scheduler | Dataflow Layer 成为协调面 |
| Rollout | trainer 调度 rollout 或 rollout pool | RaaS：rollout 作为独立服务 |
| Trainer | 既训练又常常管数据/同步/调度 | 只消费 batch、发布权重 |
| 多策略 | 通常需要额外 pipeline 工程 | 多 policy / 多 trainer 作为一等组件 |
| 数据算法 | dynamic sampling、filter、replay 往往嵌入训练逻辑 | 作为 Dataflow policy 插件 |
| 弹性/异构/跨区域 | 通常需要专门工程 | 由 RaaS + Weight Manager + Dataflow 自然承载 |



### Dataflow Layer 做了什么

AstraFlow 的 Dataflow Layer 是真正的 orchestrator。它不仅是 queue，还负责：

```text {"data-theme":"githubLight"}
selective rollout
curriculum scheduling
post-rollout filtering
dynamic sampling
replay
data mixing
staleness correction
multi-policy routing
backpressure
```

 Dataflow Layer 能根据 producing policy、model version、timestamp、reward statistics、task type 等 metadata，把 policy-specific、shared、mixed data stream 路由给不同 trainer，而不需要 trainer-to-trainer 直接协调。

### 最终效果

在 multi-policy collaborative training 中，达到与现有系统相当或更好的准确率，同时训练时间 **2\.7×** 加速。论文还展示它支持 multi-policy training、elastic scaling、heterogeneous cross-region execution、composable data algorithms，且不需要 system-level code changes。

---

## 8\. 总结

这些框架其实是在解决同一组问题，但抽象层次不同。

### 问题 A：长尾 rollout 卡住 trainer

```text {"data-theme":"githubLight"}
同步系统：
  等最慢样本 → trainer idle

AReaL：
  rollout 持续生成，trainer 收够 batch 就训练

veRL fully async：
  Rollouter 逐样本写 MessageQueue，Trainer 逐样本拉

SLIME：
  AsyncRolloutWorker 保持 in-flight generations，训练不用等最慢样本

Relax：
  TransferQueue 流式写入/消费，Actor 边到边训

AstraFlow：
  RaaS 异步生产 trajectory，Dataflow 给 trainer 提供 batch
```

### 问题 B：异步后样本变旧，影响训练效果

```text {"data-theme":"githubLight"}
AReaL：
  workload balancing + staleness-enhanced PPO

veRL：
  staleness_threshold、trigger_parameter_sync_step、require_batches、rollout old_logprob、rollout correction

Relax：
  --max-staleness 控制 off-policy drift

AstraFlow：
  Dataflow policy 做 staleness correction、fresh-first routing、block unsuitable batches

SLIME：
  fully-async rollout 页面主要强调 in-flight queue；staleness 理论和校正文档不如 veRL/AReaL 系统化
```

### 问题 C：rollout/trainer/resource 需要独立扩缩容

```text {"data-theme":"githubLight"}
AReaL：
  rollout-training fully decoupled

veRL：
  Trainer 和 Rollouter 资源隔离，分别配置节点/GPU

Relax：
  每个 role 独立 Ray Serve deployment，rollout engines 可动态增删

AstraFlow：
  RaaS 热插拔，Dataflow 不依赖固定 rollout pool

SLIME：
  更偏训练 driver + SGLang rollout，扩展重点在高性能生成和 custom rollout
```

### 问题 D：多轮 agent / env / tool use

```text {"data-theme":"githubLight"}
AReaL：
  支持 agentic runtime、multi-turn math/tool/search 等例子

veRL：
  fully async 文档包含 Multi-Turn Tool Calling；生态里也有 AgentLoop / TransferQueue / reward loop

SLIME：
  custom_generate/custom_rm 插件可接多轮 agent，文档示例提到 SWE/Codex 类 workflow

Relax：
  BaseInteractionEnv + loss masking + flexible termination，原生多轮闭环

AstraFlow：
  Env & Reward Service + RaaS + Dataflow，可支持复杂 agentic workflow 和多策略协同
```

---

## 9\.我们要做什么

```text {"data-theme":"githubLight"}

第一层：解决长尾和 GPU idle
  AReaL、veRL fully async、SLIME fully-async rollout、Relax TransferQueue 都在做

第二层：控制 staleness，尽量不伤效果
  AReaL 和 veRL 做得最系统；
  Relax 提供 max-staleness；
  AstraFlow 把 staleness 作为 dataflow policy；
  SLIME fully-async 页面更偏 rollout 工程优化

第三层：从 trainer-centered 走向更可组合系统
  Relax 用 Controller + service roles + TransferQueue
  AstraFlow 用 Dataflow Layer 作为数据控制面

第四层：多策略、多 env、弹性/异构/跨区域
  AstraFlow 抽象最完整
  Relax 生产化和多模态/多轮能力最强
  AReaL 适合 fully async reasoning/agentic RL
  veRL 适合从同步 PPO/GRPO 稳妥迁移
  SLIME 适合 Megatron + SGLang 高性能生成训练栈
```