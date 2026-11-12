MCP 서버 여는 파이썬 코드입니다.
MCP 서버에서는 LLM으로부터 유저 query를 받아 text-to-sql (로컬 LLM 사용) 파이프라인을 실행 후 LLM에게 전달하는 역할을 합니다.

uvicorn MCP.server-mod:api --host 127.0.0.1 --port 1234
으로 실행 가능합니다.

## Private configuration

See [SECURITY_SETUP.md](SECURITY_SETUP.md) before running. Private data and credentials must be supplied separately.
