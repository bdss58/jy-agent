# Browser Automation Troubleshooting

## Chrome MCP Connection Issues

### "Connection failed" or "Chrome not found"
```
Problem: Chrome MCP can't connect to Chrome DevTools
Fix: 
  1. Check if Chrome is running with remote debugging:
     run_shell("lsof -i :9222")  # Check if debug port is open
  2. If not running, the MCP server should start its own instance
  3. If using user's Chrome, it must be launched with --remote-debugging-port=9222
```

### "Tab leaked" / Too many tabs
```
Problem: Previous automation left tabs open
Fix:
  1. list_pages() → see all open tabs
  2. close_page() for each unwanted tab
  3. The agent has auto-cleanup for leaked tabs (diff-based _cleanup_leaked_tabs)
```

### "Navigation timeout"
```
Problem: Page takes too long to load
Fix:
  1. Check network: list_network_requests() → are requests pending?
  2. Try a simpler URL (e.g., the homepage instead of a deep link)
  3. Some sites block headless Chrome — try setting a realistic viewport first
```

## Anti-Bot Detection

### Cloudflare / reCAPTCHA
```
Problem: Site shows CAPTCHA or "checking your browser"
Approach:
  1. wait_for() with a longer timeout — some challenges auto-resolve
  2. take_screenshot() → show the user what's happening
  3. Ask user to solve manually if needed
  4. For Cloudflare "checking browser" — just wait 5-10 seconds
```

### "Access Denied" / 403 Responses
```
Problem: Site detects automation
Mitigations:
  1. resize_page(1440, 900) → use realistic viewport
  2. Add delays between actions (2-5 seconds)
  3. Navigate naturally (don't jump to deep URLs, go through homepage)
  4. Some sites check User-Agent — Chrome MCP uses real Chrome so this is usually fine
```

### Login Session Expired
```
Problem: Was logged in, now showing login page again
Approach:
  1. take_snapshot() → confirm it's a login page
  2. Re-do the login flow
  3. For persistent sessions, consider extracting cookies via evaluate_script
```

## Performance Tips

- **Use snapshot over screenshot** — snapshots are text (~1-5KB), screenshots are images (~100KB+)
- **Batch evaluate_script calls** — one complex JS is better than many simple ones
- **Close tabs when done** — each open tab consumes memory
- **Set viewport once** — don't resize between every action
