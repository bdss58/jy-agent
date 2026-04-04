# Anti-Detection Guide

How to reduce the automation fingerprint when using Chrome DevTools MCP
for tasks on bot-protected websites.

## Why Detection Happens

Chrome MCP uses Puppeteer under the hood, which launches Chrome with
`--enable-automation`. This sets `navigator.webdriver = true` and adds
other detectable signals. Modern anti-bot systems (Cloudflare, DataDome,
PerimeterX, Akamai) check for these.

## Detection Vectors

### 1. `navigator.webdriver` flag
- **What**: `navigator.webdriver` returns `true` in automated browsers
- **Fix**: Launch with `--disable-blink-features=AutomationControlled`

### 2. Automation Chrome flags
- **What**: `--enable-automation` adds an infobar and sets `webdriver=true`
- **Fix**: Use `--ignoreDefaultChromeArg --enable-automation` to suppress it

### 3. Headless mode artifacts
- **What**: Headless Chrome historically had different User-Agent, plugin lists,
  and WebGL renderer strings
- **Note**: Since Chrome 112+ (new headless), headful and headless are unified.
  Most headless-specific detections are obsolete. Still check `User-Agent`.

### 4. CDP detection
- **What**: Sites detect Chrome DevTools Protocol usage itself (the `Runtime`
  domain, `Page.addScriptToEvaluateOnNewDocument`, etc.)
- **Difficulty**: Hard to fully hide. CDP is the underlying protocol for all
  Puppeteer/Playwright automation.
- **Mitigation**: Minimize unnecessary CDP calls. Avoid injecting scripts.

### 5. Behavioral analysis
- **What**: Bot detection systems analyze mouse movement patterns, typing speed,
  scroll behavior, and timing between actions.
- **Mitigation**: Add realistic delays between actions. Use `type_text()` instead
  of `fill()` for more human-like typing. Vary timing.

### 6. Canvas / WebGL fingerprinting
- **What**: Automated browsers may produce different canvas/WebGL renders
- **Note**: Chrome MCP uses full Chrome, so fingerprints match real browsers.
  This is mainly an issue with older headless implementations.

## MCP Configuration for Stealth

In your MCP server config (e.g., `mcp_servers.json`), add Chrome args:

```json
{
  "chrome": {
    "command": "npx",
    "args": [
      "chrome-devtools-mcp@latest",
      "--ignoreDefaultChromeArg", "--enable-automation",
      "--chromeArg", "--disable-blink-features=AutomationControlled",
      "--chromeArg", "--disable-features=AutomationControlled"
    ]
  }
}
```

### Chrome arg format for MCP

Chrome MCP accepts args as space-separated values, NOT `--key=value` syntax:
```
✅ "--chromeArg", "--disable-blink-features=AutomationControlled"
❌ "--chromeArg=--disable-blink-features=AutomationControlled"
```

## Runtime Stealth Patches

After page load, inject stealth patches via `evaluate_script`:

```python
# Delete webdriver property
mcp__chrome__evaluate_script(
    function='''() => {
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    }'''
)

# Fix chrome.runtime (missing in automated Chrome)
mcp__chrome__evaluate_script(
    function='''() => {
        window.chrome = window.chrome || {};
        window.chrome.runtime = window.chrome.runtime || {};
    }'''
)
```

**Warning**: These patches only work AFTER page load. The site's detection
scripts may run before your patches. For pre-load injection, use
`navigate_page(initScript="...")`.

## Pre-Load Script Injection

Use `initScript` parameter on `navigate_page` to run JS before any site code:

```python
mcp__chrome__navigate_page(
    url="https://protected-site.com",
    initScript='''
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    '''
)
```

## Behavioral Mimicry Tips

1. **Add delays**: Don't click instantly after page load. Real users take time.
2. **Scroll before clicking**: Use `evaluate_script` to scroll elements into view.
3. **Type naturally**: Use `type_text()` (character-by-character) instead of `fill()` (instant).
4. **Randomize patterns**: Don't follow the exact same sequence every time.
5. **Accept cookies**: Most sites show cookie banners — dismiss them like a user would.

## Detection Test Sites

Verify your stealth setup against these:
- https://bot.sannysoft.com/ — Shows all detectable properties
- https://arh.antoinevastel.com/bots/areyouheadless — Headless detection
- https://browserleaks.com/javascript — Full JS fingerprint

## When Stealth Isn't Enough

Some sites (Cloudflare, DataDome) use advanced behavioral + TLS fingerprinting
that CDP-based tools fundamentally cannot bypass. In these cases:
- Use `web_fetch()` with cffi strategy (uses curl_cffi with Chrome TLS impersonation)
- Consider using a real browser session manually
- Use a residential proxy with proper TLS fingerprint
