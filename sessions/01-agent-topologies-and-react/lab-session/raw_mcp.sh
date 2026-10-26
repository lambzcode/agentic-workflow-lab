#!/usr/bin/env bash
# What every MCP client does, by hand: initialize, list the tools, call one.
# Start your server first (uv run python my_server.py), then:   bash raw_mcp.sh
URL=${1:-http://127.0.0.1:8000/mcp}
H=(-H "Content-Type: application/json" -H "Accept: application/json, text/event-stream")

echo "1. initialize: the server answers and gives us a session id"
SID=$(curl -s -D - -o /dev/null -X POST "$URL" "${H[@]}" -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}' \
  | tr -d '\r' | awk -F': ' 'tolower($1)=="mcp-session-id"{print $2}')
echo "   session: $SID"

curl -s -o /dev/null -X POST "$URL" "${H[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

echo; echo "2. tools/list: the names, descriptions and schemas a model gets to read"
curl -s -X POST "$URL" "${H[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

echo; echo; echo "3. resources/list and prompts/list: the two other things a server can offer"
curl -s -X POST "$URL" "${H[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","id":4,"method":"resources/list"}'
echo
curl -s -X POST "$URL" "${H[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","id":5,"method":"prompts/list"}'

echo; echo; echo "4. tools/call: run a tool"
curl -s -X POST "$URL" "${H[@]}" -H "mcp-session-id: $SID" -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_datasets","arguments":{"query":"prix des carburants"}}}'
echo
