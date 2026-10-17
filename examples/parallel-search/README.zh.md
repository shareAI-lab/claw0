[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# 可选示例：通过工具进行网页研究

学完 [s02：工具使用](../../sessions/zh/s02_tool_use.md)后，可以试试获取最新信息的工具。
这个命令行脚本通过 Streamable HTTP 连接 [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)。
`search` 查找来源和摘录，`fetch` 读取指定网址的摘录。搜索使用 Fast 模式。
两种操作都不需要 Parallel 账号或 API Key。免费匿名服务的速率限制较低，本示例不会重试请求。

十个教学章节及其依赖保持原样。请使用 Python 3.11+，在虚拟环境中单独安装本示例。
在仓库根目录执行：

```sh
python -m venv .venv
. .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r examples/parallel-search/requirements.txt

# Reuse this ID for related calls; create a new one for a new research task.
research_session=$(python -c "import uuid; print(uuid.uuid4())")
python examples/parallel-search/search.py --session-id "$research_session" search \
  "Find the latest stable Python release and its release date, using official sources" \
  --query "Python latest stable release"
python examples/parallel-search/search.py --session-id "$research_session" fetch \
  https://www.python.org/downloads/ --objective "Latest stable Python release and release date"
```

上面的会话变量使用 POSIX shell 语法。在其他 shell 中，可以直接将同一个 UUID 传给
`--session-id`，供相关调用复用；也可以省略此选项。每个新的研究任务使用新的 ID。
`search` 支持重复传入 `--query`。`fetch` 每次最多接收 20 个网址，可选的 objective 最多
200 个字符。运行 `search.py --help` 查看命令。输出包含来源网址和摘录，便于自行核对证据。
服务端工具出错时会输出错误信息并返回状态码 1；连接失败或整个调用超过 60 秒也返回 1。
Ctrl+C 可取消调用。

## 与现有 Agent 配合使用

按照根目录的[快速开始](../../README.zh.md)在同一个已激活的环境中安装章节依赖、配置模型，
然后运行 `python sessions/zh/s02_tool_use.py`，提出这样的请求：

> 用 bash 运行 `python examples/parallel-search/search.py search "Find the latest
> stable Python release using official sources" --query "Python latest stable release"`，
> 将 bash 的超时设为 90 秒。阅读返回的来源，再用同一脚本的 `fetch` 命令获取一个相关网页，
> 最后总结版本信息并附上来源链接。

这里复用 s02 的 `bash` 处理函数和 `tool_result` 消息，不会直接向模型注册 MCP 工具，
也不会为后续章节添加 MCP 支持。CLI 本身不需要模型密钥；s02 Agent 仍然需要所配置模型的
凭证。将获取的网页作为资料处理，在采信答案前核对网址和日期。

## 与 s02 的对应关系

CLI 将 `search` 分派到 `web_search`，将 `fetch` 分派到 `web_fetch`。
本地处理函数读取文件，而这里的 `ClientSession.call_tool()` 把参数发送给远程工具。
MCP SDK 负责初始化和传输，脚本输出工具的文本结果供调用方使用。每次执行都会打开并关闭
一个连接。脚本不发送任何凭证，也不读取或修改 `.env`。

运行离线调用检查：

```sh
python -m unittest discover -s examples/parallel-search -p 'test_*.py' -v
```
