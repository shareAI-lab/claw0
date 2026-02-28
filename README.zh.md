[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# claw0

**从零到一: 构建 AI Agent 网关**

> 10 个渐进式章节, 每节都是可直接运行的 Python 文件.
> 3 种语言 (英语, 中文, 日语) -- 代码 + 文档同目录.

---

## 这是什么?

大多数 Agent 教程停在"调一次 API"就结束了. 这个仓库从那个 while 循环开始, 一路带你到生产级网关.

逐章节构建一个最小化 AI Agent 网关. 10 个章节, 10 个核心概念, 约 7,000 行 Python. 每节只引入一个新概念, 前一节的代码原样保留. 学完全部 10 节, 你就能顺畅地阅读 OpenClaw 的生产代码.

```sh
s01: Agent Loop           -- 基础: while + stop_reason
s02: Tool Use             -- 让模型能调工具: dispatch table
s03: Sessions & Context   -- 会话持久化, 上下文溢出处理
s04: Channels             -- Telegram + 飞书: 完整通道管线
s05: Gateway & Routing    -- 5 级绑定, 会话隔离
s06: Intelligence         -- 灵魂, 记忆, 技能, 提示词组装
s07: Heartbeat & Cron     -- 主动型 Agent + 定时任务
s08: Delivery             -- 可靠消息队列 + 退避
s09: Resilience           -- 3 层重试洋葱 + 认证轮换
s10: Concurrency          -- 命名队列车道序列化混沌
```

## 架构概览

```
+------------------- claw0 layers -------------------+
|                                                     |
|  s10: Concurrency  (命名车道, generation 追踪)      |
|  s09: Resilience   (认证轮换, 溢出压缩)             |
|  s08: Delivery     (预写队列, 退避)                 |
|  s07: Heartbeat    (Lane 锁, cron 调度)             |
|  s06: Intelligence (8 层提示词, 混合记忆检索)       |
|  s05: Gateway      (WebSocket, 5 级路由)            |
|  s04: Channels     (Telegram 管线, 飞书 webhook)    |
|  s03: Sessions     (JSONL 持久化, 3 阶段重试)       |
|  s02: Tools        (dispatch table, 4 个工具)       |
|  s01: Agent Loop   (while True + stop_reason)       |
|                                                     |
+-----------------------------------------------------+
```

## 章节依赖关系

```
s01 --> s02 --> s03 --> s04 --> s05
                 |               |
                 v               v
                s06 ----------> s07 --> s08
                 |               |
                 v               v
                s09 ----------> s10
```

- s01-s02: 基础 (无依赖)
- s03: 基于 s02 (为工具循环添加持久化)
- s04: 基于 s03 (通道产生 InboundMessage 给会话)
- s05: 基于 s04 (将通道消息路由到 Agent)
- s06: 基于 s03 (使用会话做上下文, 添加提示词层)
- s07: 基于 s06 (心跳使用灵魂/记忆构建提示词)
- s08: 基于 s07 (心跳输出经由投递队列)
- s09: 基于 s03+s06 (复用 ContextGuard 做溢出层, 模型配置)
- s10: 基于 s07 (将单一 Lock 替换为命名车道系统)

## 快速开始

```sh
# 1. 克隆并进入目录
git clone https://github.com/shareAI-lab/claw0.git && cd claw0

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置
cp .env.example .env
# 编辑 .env: 填入 ANTHROPIC_API_KEY 和 MODEL_ID

# 4. 运行任意章节 (选择你的语言)
python sessions/zh/s01_agent_loop.py    # 中文
python sessions/en/s01_agent_loop.py    # English
python sessions/ja/s01_agent_loop.py    # Japanese
```

## 学习路径

每节只加一个新概念, 上一节的代码完整保留:

```
Phase 1: 基础         Phase 2: 连接            Phase 3: 智能            Phase 4: 自治           Phase 5: 生产
+----------------+    +-------------------+    +-----------------+     +-----------------+    +-----------------+
| s01: Loop      |    | s03: Sessions     |    | s06: Intelligence|    | s07: Heartbeat  |    | s09: Resilience |
| s02: Tools     | -> | s04: Channels     | -> |   灵魂, 记忆,   | -> |     & Cron       | -> |   & Concurrency |
|                |    | s05: Gateway      |    |   技能, 提示词   |    | s08: Delivery   |    | s10: Lanes      |
+----------------+    +-------------------+    +-----------------+     +-----------------+    +-----------------+
 循环 + dispatch       持久化 + 路由             人格 + 回忆             主动行为 + 可靠投递      重试 + 序列化
```

## 章节详情

| # | 章节 | 核心概念 | 行数 |
|---|------|---------|------|
| 01 | Agent Loop | `while True` + `stop_reason` -- 这就是一个 Agent | ~175 |
| 02 | Tool Use | 工具 = schema dict + handler map. 模型选名字, 你查表执行 | ~445 |
| 03 | Sessions | JSONL: 写入追加, 读取重放. 太大了? 总结旧消息 | ~890 |
| 04 | Channels | 每个平台都不同, 但最终都生产同一个 `InboundMessage` | ~780 |
| 05 | Gateway | 绑定表将 (channel, peer) 映射到 agent. 最具体的匹配胜出 | ~625 |
| 06 | Intelligence | 系统提示词 = 磁盘上的文件. 换文件, 换人格, 不改代码 | ~750 |
| 07 | Heartbeat & Cron | 定时线程: "该不该跑?" + 和用户消息共用同一管线 | ~660 |
| 08 | Delivery | 先写磁盘, 再尝试发送. 崩溃也丢不了消息 | ~870 |
| 09 | Resilience | 3 层重试洋葱: 认证轮换, 溢出压缩, 工具循环 | ~1130 |
| 10 | Concurrency | 命名车道 + FIFO 队列, generation 追踪, Future 返回 | ~900 |

## 仓库结构

```
claw0/
  README.md              English README
  README.zh.md           Chinese README
  README.ja.md           Japanese README
  .env.example           配置模板
  requirements.txt       Python 依赖
  sessions/              所有教学章节 (代码 + 文档)
    en/                  English
      s01_agent_loop.py  s01_agent_loop.md
      s02_tool_use.py    s02_tool_use.md
      ...                (10 .py + 10 .md)
    zh/                  中文
      s01_agent_loop.py  s01_agent_loop.md
      ...                (10 .py + 10 .md)
    ja/                  Japanese
      s01_agent_loop.py  s01_agent_loop.md
      ...                (10 .py + 10 .md)
  workspace/             共享工作区样例
    SOUL.md  IDENTITY.md  TOOLS.md  USER.md
    HEARTBEAT.md  BOOTSTRAP.md  AGENTS.md  MEMORY.md
    CRON.json
    skills/example-skill/SKILL.md
```

每个语言文件夹自包含: 可运行的 Python 代码 + 配套文档. 代码逻辑跨语言一致, 注释和文档因语言而异.

## 前置要求

- Python 3.11+
- Anthropic (或兼容服务商) 的 API key

## 依赖

```
anthropic>=0.39.0
python-dotenv>=1.0.0
websockets>=12.0
croniter>=2.0.0
python-telegram-bot>=21.0
httpx>=0.27.0
```

## 相关项目

- **[learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)** -- 姊妹教学仓库, 用 12 个递进课程从零构建一个智能体**框架** (nano Claude Code)。claw0 聚焦于网关路由、多通道接入和主动行为, learn-claude-code 则深入智能体的内部设计: 结构化规划 (TodoManager + nag)、上下文压缩 (三层 compact)、基于文件的任务持久化与依赖图、团队协调 (JSONL 邮箱、关机/计划审批 FSM)、自治式自组织, 以及 git worktree 隔离的并行执行。如果你想理解一个生产级单元智能体的内部运作, 从那里开始。

## 许可证

MIT
