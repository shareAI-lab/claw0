[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# オプション例：ツールで Web を調べる

[s02：ツール使用](../../sessions/ja/s02_tool_use.md)の後に、最新情報を取得するツールを試せます。
この CLI は Streamable HTTP で [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
に接続します。`search` は情報源と抜粋を検索し、`fetch` は指定した URL の抜粋を取得します。
検索は Fast モードを使用します。どちらも Parallel のアカウントや API キーは不要です。
無料の匿名サービスには低めのレート制限があり、この例はリクエストを再試行しません。

10 のレッスンとその依存関係はそのままです。この例は Python 3.11+ の仮想環境で
別途インストールしてください。リポジトリのルートで実行します：

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

上のセッション変数は POSIX シェルの構文です。他のシェルでは、関連する呼び出しで同じ
UUID を直接 `--session-id` に渡してください。このオプションは省略もできます。
新しい調査には新しい ID を使います。`search` は複数の `--query` を受け付けます。
`fetch` は最大 20 個の URL と、最大 200 文字の任意の objective を受け付けます。
コマンドは `search.py --help` で確認できます。出力には情報源の URL と抜粋が含まれ、
根拠を自分で確認できます。サーバーのツールエラーはメッセージと終了コード 1 を返します。
接続エラーや全体で 60 秒のタイムアウトも終了コード 1 を返します。Ctrl+C で中断できます。

## 既存のエージェントから使う

同じ有効な仮想環境で、ルートの[クイックスタート](../../README.ja.md)に従ってレッスンの
依存関係をインストールし、モデルを設定します。`python sessions/ja/s02_tool_use.py` を
実行し、次のように依頼します：

> bash で `python examples/parallel-search/search.py search "Find the latest
> stable Python release using official sources" --query "Python latest stable release"` を
> 実行してください。bash のタイムアウトは 90 秒にしてください。返された情報源を読み、
> 同じスクリプトの `fetch` コマンドで関連する URL を取得し、出典リンク付きでバージョンを要約してください。

s02 の既存の `bash` ハンドラーと `tool_result` メッセージを使います。
MCP ツールをモデルへ直接登録したり、後のレッスンに MCP サポートを追加したりはしません。
CLI 自体にモデルのキーは不要ですが、s02 エージェントには設定したモデルの認証情報が必要です。
取得したページは参考資料として扱い、回答を信頼する前に URL と日付を確認してください。

## s02 との対応

CLI は `search` を `web_search` に、`fetch` を `web_fetch` に振り分けます。
ローカルのハンドラーがファイルを読む代わりに、`ClientSession.call_tool()` が引数を
リモートツールへ送ります。MCP SDK が初期化と通信を担当し、スクリプトがツールの
テキスト結果を出力します。実行ごとに接続を開いて閉じます。認証情報は送信せず、
`.env` の読み取りや変更もしません。

オフラインの呼び出しチェック：

```sh
python -m unittest discover -s examples/parallel-search -p 'test_*.py' -v
```
