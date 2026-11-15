#!/bin/bash
# 1 if the vLLM server on port $1 answers a tiny chat request within 120 s (the /v1/models endpoint can stay up while the
# engine is hung, 2026-11-13)
M=$(curl -s --noproxy '*' -m 5 http://127.0.0.1:$1/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
[ -n "$M" ] && curl -s --noproxy '*' -m 120 http://127.0.0.1:$1/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":5}" | grep -q '"choices"'
