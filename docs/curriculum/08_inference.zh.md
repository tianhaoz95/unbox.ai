# 8 · 推理引擎

## 目标

推理引擎的目标是**在架构上与 SGLang/vLLM 对齐**，而非在性能上追平。比生产系统慢至多 50% 是可接受的 —— 生产级引擎中存在的每个子系统都必须在这里以可读性优先的方式实现。

## 架构：预填充-解码分离

引擎将 LLM 推理的两个阶段拆分为独立的工作进程池：

```
HTTP 客户端
    │
    ▼
FastAPI 前端  (server.py)
    │  ZMQ PUSH/PULL
    ▼
路由器  (router.py)
    │                     │
    ▼                     ▼
预填充工作进程         解码工作进程
（计算密集型）         （内存带宽密集型）
    │  KV 缓存传输（ZMQ）
    └─────────────────────▶│
                           ▼
                   Token 流 → 路由器 → 客户端
```

**为什么要分离？**

- **预填充**并行处理整个提示词 —— 对 T 个 Token 做一次前向传播，是计算密集型（GPU 利用率 ≈ 100%）。
- **解码**每步生成一个 Token，持续访问增长中的 KV 缓存，是内存带宽密集型（GPU 利用率 ≈ 10–20%）。

将两者混在同一 GPU 上，内存受限的解码步骤会浪费 GPU 的计算能力。分离的工作进程池允许各阶段独立扩展。

## KV 缓存：分页分配

为每个序列存储一个连续的 KV 缓存缓冲区，在序列长度不同时会浪费 GPU 内存。vLLM 引入的分页方案将 KV 缓存视为虚拟内存：

- 预先分配一个固定大小**块**（例如 16 Token × num_kv_heads × head_dim）的池
- 每个序列有一张**块表**，将逻辑位置映射到物理块索引
- 完成的序列将其块归还给空闲列表

**基数树**位于块分配器之上，用于**前缀缓存** —— 如果两个请求共享相同的提示词前缀，则共享同一批 KV 块（无需重新计算）。

```
空闲列表：[block 0, block 3, block 7, ...]

序列 A：[block 1][block 4][block 9]
序列 B：[block 1][block 4][block 2]  ← 与 A 共享前缀块
```

## 调度器：连续批处理

没有连续批处理，引擎要么每次只服务一个请求，要么等待凑满一个批次 —— 两种方式都会浪费吞吐量。连续批处理允许新请求在现有序列完成时立即加入批次：

1. 每步，调度器填充一个批次：新请求的预填充 Token + 每个在途序列的一个解码 Token
2. 前向传播后，已完成的序列（触发 EOS 或达到 `max_tokens`）被移出
3. 新请求立即占据空出的槽位

**分块预填充**将长提示词拆分到多个步骤中，防止单个大型预填充阻塞解码队列。

## 张量并行：NCCL

在每个工作进程池内，GPU 间分割模型权重：

- **列并行**：Q、K、V、gate、up 投影沿输出维度分割到多个 GPU
- **行并行**：O、down 投影沿输入维度分割，之后进行 NCCL All-Reduce

这与训练中使用的 Megatron 风格张量并行相同，在推理时同样适用。

## 进程间通信：ZMQ

所有进程间通信使用 ZMQ：

- **前端 → 路由器**：PUSH/PULL 套接字对，每个请求一条消息
- **路由器 → 预填充工作进程**：将请求分发到预填充进程池
- **预填充 → 解码**：预填充完成后的序列化 KV 块传输
- **解码 → 路由器**：每生成一个 Token 时的 Token 流

简单优先：无 RDMA，无零拷贝，仅 ZMQ 字节传输。

## Triton 内核

自定义算子位于 `unbox_platform/infer/kernels/`，以 **Triton**（而非 CUDA）编写。

### 分页解码注意力

关键内核：单个 Token 的 Q 对块表 KV 缓存做注意力计算。PyTorch 的 `scaled_dot_product_attention` 无法表达这种操作，因为它没有块表的概念。

```
Q: [1, num_heads, head_dim]
KV 块: [num_blocks, 2, block_size, num_kv_heads, head_dim]
块表: [seq_len // block_size]
→ 注意力输出: [1, num_heads, head_dim]
```

### 融合 RMSNorm

将 RMS 计算和权重乘法融合到一个内核中 —— 消除每个归一化层的一次内存往返。

### 融合 SwiGLU

将 `silu(gate) * up` 融合到一个内核中，节省中间激活张量的一次写入和读取。

### 融合 RoPE

在注意力之前将旋转位置编码原地应用于 Q 和 K，节省完整激活张量的一次写入+读取。

!!! info "预填充使用 Flash Attention"
    预填充使用标准的 `F.scaled_dot_product_attention(is_causal=True)`，通过 PyTorch 的 C++ 后端调度到 Flash Attention。预填充无需自定义内核。

## 组件映射

```
unbox_platform/infer/
  kernels/
    paged_attention.py   # Triton 分页解码注意力
    rms_norm.py          # Triton 融合 RMSNorm
    swiglu.py            # Triton 融合 SwiGLU
    rope.py              # Triton 融合 RoPE
  kvcache.py             # 块分配器、基数树、LRU 淘汰
  scheduler.py           # 每个工作进程的连续批处理循环
  engine.py              # 单工作进程前向传播
  worker.py              # 预填充/解码工作进程入口点
  router.py              # 请求分发器，KV 传输协调器
  server.py              # FastAPI：POST /v1/chat/completions，GET /v1/models
  distributed.py         # NCCL 张量并行设置
  messaging.py           # ZMQ 套接字抽象
  sampling.py            # 贪心、top-k、top-p、温度采样
  config.py              # InferConfig 数据类
```

## 参考

KV 缓存设计（带 LRU 淘汰的基数树）和预填充/解码调度器分离方案参考自 [mini-sglang](https://github.com/sgl-project/mini-sglang)。Mini-sglang 将注意力委托给 FlashInfer —— 我们改为自己编写 Triton 内核。

## 当前状态

规划中。实现将在预训练和 SFT 验证之后开始。
