[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# claw0

**从零到一: 构建一个类 OpenClaw 的 AI Agent 网关**

> 10 个渐进式章节, 每节引入一个核心机制.
> 每个章节都是可直接运行的 Python 文件.

---

## 这是什么?

这是一个教学仓库, 带你从零开始构建一个最小化的 AI Agent 网关, 灵感来自 [OpenClaw](https://github.com/openclaw/openclaw) 架构. 每个章节只添加一个机制, 不改变核心循环.

```
s01: Agent Loop        -- 基础: while + stop_reason
s02: Tool Use          -- 赋予模型行动能力: dispatch map
s03: Sessions          -- 跨重启的持久化对话
s04: Multi-Channel     -- 同一个大脑, 多个出口
s05: Gateway Server    -- 交换机: WebSocket + JSON-RPC
s06: Routing           -- 每条消息都找到归属
s07: Soul & Memory     -- 赋予灵魂, 让它记住
s08: Heartbeat         -- 不只是被动响应, 还能主动出击
s09: Cron Scheduler    -- 在正确的时间做正确的事
s10: Delivery Queue    -- 消息永不丢失
```

## 架构概览

```
+--------- claw0 architecture ---------+
|                                           |
|  s10: Delivery Queue (reliable delivery)  |
|  s09: Cron Scheduler (timed tasks)        |
|  s08: Heartbeat (proactive behavior)      |
|  s07: Soul & Memory (personality + recall)|
|  s06: Routing (multi-agent binding)       |
|  s05: Gateway (WebSocket/HTTP server)     |
|  s04: Multi-Channel (channel plugins)     |
|  s03: Sessions (persistent state)         |
|  s02: Tools (bash/read/write/edit)        |
|  s01: Agent Loop (while + stop_reason)    |
|                                           |
+-------------------------------------------+
```

## 快速开始

```sh
# 1. 克隆并进入目录
git clone https://github.com/shareAI-lab/claw0.git && cd claw0

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置
cp .env.example .env
# 编辑 .env, 填入你的 API key 和模型名称

# 4. 运行任意章节
python agents/s01_agent_loop.py
python agents/s02_tool_use.py
# ... 以此类推
```

## 学习路径

```
Phase 1: THE LOOP       Phase 2: STATE        Phase 3: GATEWAY      Phase 4: INTELLIGENCE  Phase 5: OPERATIONS
+----------------+      +----------------+    +----------------+    +----------------+     +----------------+
| s01: Agent Loop|      | s03: Sessions  |    | s05: Gateway   |    | s07: Soul/Mem  |     | s09: Cron      |
| s02: Tool Use  | ---> | s04: Multi-Ch  | -> | s06: Routing   | -> | s08: Heartbeat | --> | s10: Delivery  |
| (0 -> 1 tools) |      | (state+channel)|    | (server+route) |    | (persona+auto) |     | (schedule+rely)|
+----------------+      +----------------+    +----------------+    +----------------+     +----------------+
    2 tools                2 mechanisms           2 mechanisms          2 mechanisms           2 mechanisms
```

## 章节详情

| # | Section | 格言 | 核心机制 | 新增概念 |
|---|---------|------|----------|----------|
| 01 | Agent Loop | "一个循环统治一切" | while + stop_reason | LLM API, 消息历史 |
| 02 | Tool Use | "赋予模型行动能力" | TOOL_HANDLERS dispatch | 工具 schema, 安全执行 |
| 03 | Sessions | "跨重启的持久化对话" | SessionStore + JSONL | 持久化, session key |
| 04 | Multi-Channel | "同一个大脑, 多个出口" | Channel 插件接口 | 抽象层, 消息标准化 |
| 05 | Gateway Server | "交换机" | WebSocket + JSON-RPC | 服务端架构, RPC |
| 06 | Routing | "每条消息都找到归属" | Binding resolution | 多 Agent, 路由优先级 |
| 07 | Soul & Memory | "赋予灵魂, 让它记住" | SOUL.md + MemoryStore | 人格设定, 向量搜索 |
| 08 | Heartbeat | "不只是被动响应, 还能主动出击" | HeartbeatRunner | 自主行为 |
| 09 | Cron Scheduler | "在正确的时间做正确的事" | CronService + 3 种调度类型 | at/every/cron, 自动禁用 |
| 10 | Delivery Queue | "消息永不丢失" | DeliveryQueue + backoff | 至少一次投递, 磁盘持久化 |

## OpenClaw 对比

| 概念 | claw0 (教学版) | OpenClaw (生产版) |
|------|----------------|-------------------|
| Agent Loop | 简单 while 循环 | 基于 Lane 的并发, 重试洋葱模型 |
| Tools | 4 个基础工具 | 50+ 工具, 含安全策略 |
| Sessions | JSON 文件 | JSONL 转录 + sessions.json 元数据 |
| Channels | CLI + 文件模拟 | Telegram, Discord, Slack, Signal, WhatsApp 等 15+ 渠道 |
| Gateway | websockets 库 | 原生 http + ws, 插件 HTTP 路由 |
| Routing | 优先级绑定 | 多层级: peer/guild/team/account/channel + 身份关联 |
| Memory | 关键词搜索 | SQLite-vec + FTS5 + embedding 缓存 |
| Heartbeat | Thread + timer | 6 步检查链, Lane 互斥, 24h 去重 |
| Cron | 3 种调度类型 (at/every/cron) | 完整 cron 解析器, 时区支持, SQLite 运行日志 |
| Delivery | 文件队列 + backoff | SQLite 队列, jitter, 优先级, 批量投递 |

## 文档结构

```
docs/
  en/    -- English documentation
  zh/    -- Chinese documentation
  ja/    -- Japanese documentation
```

## 前置要求

- Python 3.11+
- Anthropic (或兼容服务商) 的 API key

## 许可证

MIT - 可自由用于学习和教学.
