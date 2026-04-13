# Chrome & MCP

## Chrome DevTools
- Chrome 146+: requires non-default --user-data-dir for remote debugging; silently ignores --remote-debugging-port on default profile
- Chrome MCP launches its own instance via Puppeteer pipe (no permission popups)
- Chrome is single-instance on macOS — wrapper apps don't work if Chrome already running
- Chrome cookies encrypted with macOS Keychain — can't copy to independent profile

## Anti-bot / Automation
- Puppeteer's --enable-automation sets navigator.webdriver=true
- Fix with --ignoreDefaultChromeArg --enable-automation and --chromeArg --disable-blink-features=AutomationControlled
- Chrome MCP args are space-separated arrays, not =syntax

## MCP Integration
- `mcp(action='connect', server='chrome')` registers `mcp__chrome__*` tools directly into the main registry
- `jyagent/tools/mcp_tool.py` is only a control tool; actual Chrome actions are native registered tools after connect
- `MCPManager._call_mcp_tool()` auto-connects missing servers and retries once after forced reconnect on dead-browser errors
- `chrome_ensure_connected()` health-checks Chrome with `list_pages` because the stdio pipe can stay alive after the browser/CDP connection is dead
- Keepalive pings alone are not enough to prove Chrome is healthy
- MCP SDK v1.26: redirect subprocess stderr to log file (fixes "Assertion failed" in CLI)
- Google search works from this environment — past failures were MCP connection issues, not network blocks

## Operational Gotchas
- Stale Chrome sessions can return near-empty snapshots after long idle periods; reconnect instead of trusting the existing session
- Web fetch's Chrome tier uses deterministic MCP calls and cleans up temporary tabs; if it connected Chrome itself, it disconnects afterward
- For search result pages, JS extraction via `evaluate_script` is often better than raw `take_snapshot` because it can recover real result URLs
