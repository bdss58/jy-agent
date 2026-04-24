# GFW Fallback Proxy Workflow

**One-line trigger in MEMORY.md.** Step-by-step here.

## When to use
A network operation fails or times out against a likely-blocked host:
- `ghcr.io`
- `raw.githubusercontent.com`
- `huggingface.co`
- pypi / GitHub LFS
- `docker.io` pulls
- Google APIs

**Always try direct first** (often works). Proxy only on failure.

## Steps

### 1. Check the SOCKS listener
```bash
lsof -iTCP:1080 -sTCP:LISTEN -nP
```

### 2. If missing, start the SSH tunnel
The host alias is `thost` in `~/.ssh/config`. Use `run_background` (NOT
`run_shell` — it's long-lived):
```bash
ssh -NL 1080:localhost:1080 thost
```

### 3. Retry through SOCKS5

| Tool | Flag |
|---|---|
| `curl` | `--socks5-hostname 127.0.0.1:1080` |
| `git`  | `-c http.proxy=socks5h://127.0.0.1:1080` |
| env    | `ALL_PROXY=socks5h://127.0.0.1:1080 HTTPS_PROXY=socks5h://127.0.0.1:1080` |
| `pip`  | `--proxy socks5h://127.0.0.1:1080` |

## Critical: `socks5h` not `socks5`

Always use `socks5h://` (remote DNS resolution) — never plain `socks5://`
(local DNS). Local DNS leaks to the local resolver and risks DNS poisoning,
defeating the point of the tunnel.
