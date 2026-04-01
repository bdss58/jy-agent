---
name: browser-automation
description: >-
  Automate Chrome browser interactions using MCP Chrome DevTools Protocol. Use this skill
  whenever the user asks to navigate websites, fill forms, click buttons, take screenshots,
  scrape dynamic content, debug web pages, test web UIs, log into sites, or interact with
  web applications. TRIGGER on: "open this page", "click the button", "fill the form",
  "scrape", "screenshot", browser testing, Taobao/Zhihu/any website interaction, login
  automation, web scraping, DOM inspection, cookie extraction, network monitoring.
  DO NOT TRIGGER on: static file reading, API-only requests (use web_fetch), or CLI tools.
metadata:
  author: jy-agent
  version: "2.0"
---

# Browser Automation

Automate Chrome browser interactions via MCP Chrome DevTools.

## Decision Tree: Choose Your Approach

```
User task → Does it need a real browser?
├─ No (just fetch HTML/API) → Use web_fetch tool directly, not this skill
└─ Yes → Is Chrome MCP connected?
    ├─ No → mcp(action="connect", server="chrome")
    │        └─ Failed? → Check references/troubleshooting.md
    └─ Yes → What kind of task?
        ├─ Simple page read → Navigate → Snapshot → Extract
        ├─ Form fill/login → Navigate → Snapshot → Fill → Click → Verify
        ├─ Multi-step workflow → Use the Snapshot-Act-Verify loop (below)
        └─ Scrape dynamic content → Navigate → Wait → Evaluate JS
```

## Core Pattern: Snapshot → Act → Verify

Every browser interaction follows this loop. **Never skip the snapshot.**

```
1. navigate_page(url)           # Go to the page
2. take_snapshot()              # See the a11y tree with UIDs — REQUIRED before any action
3. <action>(uid from snapshot)  # click, fill, type using UIDs
4. take_snapshot() or           # Verify the result
   take_screenshot()
```

**Why snapshot first?** The accessibility tree gives you stable UIDs for elements. Without it, you're guessing at selectors that may not exist or may have changed.

## Actions Reference

| Action | Tool | When to use |
|--------|------|-------------|
| Navigate | `navigate_page(url)` | Go to a URL |
| See structure | `take_snapshot()` | Get a11y tree with UIDs (preferred — low tokens) |
| See visually | `take_screenshot()` | Get a PNG for visual verification |
| Click | `click(uid)` | Buttons, links, checkboxes |
| Fill text | `fill(uid, value)` | Text inputs, textareas (sets value directly) |
| Type text | `type_text(uid, text)` | When you need keystrokes (autocomplete, search) |
| Keyboard | `press_key(key)` | "Enter", "Tab", "Escape", "Control+A" |
| Wait | `wait_for(text)` | Wait for text to appear after navigation |
| Run JS | `evaluate_script(js)` | Complex DOM ops, data extraction, scroll |
| New tab | `new_page(url)` | Open a new tab |
| Switch tab | `select_page(index)` | Switch between open tabs |
| List tabs | `list_pages()` | See all open tabs |
| Close tab | `close_page()` | Clean up tabs when done |
| Resize | `resize_page(w, h)` | Set viewport (default 1440×900) |
| Console | `list_console_messages()` | Debug JS errors |
| Network | `list_network_requests()` | Debug API calls |

## Common Workflows

See [📋 Common Workflows](references/workflows.md) for step-by-step patterns:
- Login to a website
- Scrape a data table
- Fill a multi-page form
- Handle pagination
- Download files

## Anti-Patterns

❌ **Don't** try to click/fill without taking a snapshot first
✅ **Do** always snapshot → find UID → act

❌ **Don't** navigate rapidly between pages without waiting
✅ **Do** use `wait_for(text)` after navigation to confirm page loaded

❌ **Don't** use `type_text` for filling form fields (slow, unreliable)
✅ **Do** use `fill(uid, value)` for form fields; `type_text` only for search/autocomplete

❌ **Don't** take screenshots for structural analysis (wastes tokens)
✅ **Do** use `take_snapshot()` for structure, `take_screenshot()` only for visual verification

❌ **Don't** leave tabs open after finishing a task
✅ **Do** close tabs with `close_page()` to prevent tab leaks

❌ **Don't** assume page structure from memory or training data
✅ **Do** always inspect the actual page — sites change their DOM frequently

## Anti-Bot & Stealth Tips

See [🛡️ Anti-Detection Guide](references/anti-detection.md) for site-specific strategies.

Quick rules:
- Set viewport to realistic size: `resize_page(1440, 900)`
- Don't navigate more than 1 page/second
- For Chinese e-commerce (Taobao, JD): expect CAPTCHAs, use delays
- For sites behind Cloudflare: the Chrome MCP tier already uses a real browser, which helps
- If blocked, try adding realistic delays between actions (1-3 seconds)
