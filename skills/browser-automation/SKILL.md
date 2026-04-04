---
name: browser-automation
description: >-
  Automate browser interactions using Chrome DevTools MCP tools. Use this skill
  whenever the user asks to interact with a web page, click buttons, fill forms,
  take screenshots, test a website, scrape page content, debug frontend issues,
  inspect network requests, analyze page performance, run Lighthouse audits, or
  automate any browser-based workflow. TRIGGER on: "click", "fill the form",
  "navigate to", "take a screenshot", "test the page", "check the website",
  "scrape", "inspect network", "debug frontend", "performance trace",
  "Lighthouse audit", "open the browser", "browser automation", "check console
  errors", "test mobile", "emulate", "wait for element", any task requiring
  interaction with a live web page. DO NOT TRIGGER on: web search queries
  (web-search skill), fetching a known URL for content (web_fetch), git
  operations, code review without browser context, or pure API testing.
metadata:
  author: jy-agent
  version: "1.0"
---

# Browser Automation

Automate browser interactions via Chrome DevTools MCP — navigate pages, interact
with elements, debug frontend issues, analyze performance, and inspect network traffic.

> **Prerequisite**: Chrome MCP must be connected. If not, run:
> ```python
> mcp(action="connect", server="chrome")
> ```

## Decision Tree: What Are You Doing?

```
User request →
├─ Navigate / interact with a page
│   ├─ Simple page visit → Quick Navigation Flow
│   ├─ Fill form / click buttons → Form & Interaction Flow
│   ├─ Multi-step workflow (checkout, signup) → Multi-Page Flow
│   └─ File upload → Upload Flow
│
├─ Inspect / debug
│   ├─ Console errors → Console Debugging Flow
│   ├─ Network requests / API failures → Network Debugging Flow
│   ├─ DOM / element inspection → Snapshot & Script Flow
│   └─ Visual verification → Screenshot Flow
│
├─ Performance / audits
│   ├─ Core Web Vitals / speed → Performance Trace Flow
│   ├─ Accessibility / SEO / best practices → Lighthouse Flow
│   └─ Memory leaks → Memory Snapshot Flow
│
├─ Emulation / responsive testing
│   ├─ Mobile viewport → Emulation Flow
│   ├─ Slow network (3G/4G) → Emulation Flow
│   └─ Dark mode / geolocation → Emulation Flow
│
└─ Scraping / data extraction
    ├─ Structured data from page → Snapshot + Script Flow
    └─ Multiple pages → Multi-Page Scraping Flow
```

## Step 0: Ensure Chrome MCP is Connected

```python
# Always verify connection first
mcp(action="status")
# If chrome is "not connected":
mcp(action="connect", server="chrome")
```

Anti-detection tip: Chrome MCP launches via Puppeteer with `--enable-automation`.
For sites with bot detection, the MCP config should include flags to reduce
detection surface. See [references/anti-detection.md](references/anti-detection.md).

## Core Flows

### Quick Navigation

```python
# Open a new page
mcp__chrome__new_page(url="https://example.com")

# Or navigate an existing page
mcp__chrome__navigate_page(url="https://example.com")

# Take a snapshot to see page structure (preferred over screenshot)
mcp__chrome__take_snapshot()

# Take a screenshot for visual verification
mcp__chrome__take_screenshot()
```

**Key principle**: Always `take_snapshot()` after navigation. Snapshots return
the accessibility tree with UIDs for every interactive element. You need UIDs
to click, fill, hover, etc.

### Form & Interaction Flow

```python
# 1. Take snapshot to find element UIDs
mcp__chrome__take_snapshot()

# 2. Fill a single field
mcp__chrome__fill(uid="<element-uid>", value="hello@example.com")

# 3. Fill multiple fields at once (more efficient)
mcp__chrome__fill_form(elements=[
    {"uid": "name-uid", "value": "John Doe"},
    {"uid": "email-uid", "value": "john@example.com"},
    {"uid": "country-uid", "value": "United States"}
])

# 4. Click submit
mcp__chrome__click(uid="<submit-button-uid>")

# 5. Wait for result
mcp__chrome__wait_for(text=["Success", "Thank you"])
```

### Multi-Page Flow

For multi-step workflows (checkout, wizards, onboarding):

```python
# Step 1: Navigate to start
mcp__chrome__navigate_page(url="https://shop.example.com/cart")
mcp__chrome__take_snapshot()

# Step 2: Interact with current page
mcp__chrome__click(uid="<checkout-btn-uid>")
mcp__chrome__wait_for(text=["Shipping Address"])
mcp__chrome__take_snapshot()  # RE-SNAPSHOT after each navigation!

# Step 3: Fill next form
mcp__chrome__fill_form(elements=[...])
mcp__chrome__click(uid="<continue-uid>")

# Repeat: wait → snapshot → interact → wait → snapshot
```

**Critical**: Re-take snapshot after every page navigation or significant DOM
change. UIDs from old snapshots become stale.

### Upload Flow

```python
mcp__chrome__take_snapshot()
mcp__chrome__upload_file(uid="<file-input-uid>", filePath="/path/to/file.pdf")
```

### Console Debugging Flow

```python
# List all console messages (errors, warnings, logs)
mcp__chrome__list_console_messages()

# Filter to just errors
mcp__chrome__list_console_messages(types=["error", "warn"])

# Get details on a specific message
mcp__chrome__get_console_message(msgid=42)
```

### Network Debugging Flow

```python
# List all network requests since last navigation
mcp__chrome__list_network_requests()

# Filter by type (xhr, fetch, document, etc.)
mcp__chrome__list_network_requests(resourceTypes=["xhr", "fetch"])

# Inspect a specific request (headers, body, response)
mcp__chrome__get_network_request(reqid=15)

# Save response body to file for analysis
mcp__chrome__get_network_request(reqid=15, responseFilePath="response.json")
```

### Snapshot & Script Flow

For DOM inspection or data extraction:

```python
# Accessibility tree snapshot (structured, has UIDs)
mcp__chrome__take_snapshot()

# Verbose snapshot (all a11y properties)
mcp__chrome__take_snapshot(verbose=True)

# Execute custom JavaScript for complex extraction
mcp__chrome__evaluate_script(
    function='() => { return document.querySelectorAll("h2").length }'
)

# With element arguments (pass UIDs)
mcp__chrome__evaluate_script(
    function='(el) => { return el.getBoundingClientRect() }',
    args=["<element-uid>"]
)
```

### Screenshot Flow

```python
# Viewport screenshot
mcp__chrome__take_screenshot()

# Full page screenshot
mcp__chrome__take_screenshot(fullPage=True)

# Element screenshot
mcp__chrome__take_screenshot(uid="<element-uid>")

# Save to file instead of inline
mcp__chrome__take_screenshot(filePath="screenshot.png")

# JPEG with quality control (smaller files)
mcp__chrome__take_screenshot(format="jpeg", quality=80)
```

### Performance Trace Flow

```python
# 1. Navigate to the page first
mcp__chrome__navigate_page(url="https://example.com")

# 2. Start trace with auto-reload and auto-stop
mcp__chrome__performance_start_trace(reload=True, autoStop=True)

# 3. Analyze insights from the trace
# The trace results show available insight sets with IDs
mcp__chrome__performance_analyze_insight(
    insightSetId="<id-from-results>",
    insightName="LCPBreakdown"
)

# Save raw trace data for external analysis
mcp__chrome__performance_start_trace(
    reload=True, autoStop=True,
    filePath="trace.json.gz"
)
```

Key insights to analyze: `LCPBreakdown`, `CLSCulprits`, `DocumentLatency`,
`RenderBlocking`, `ThirdParties`, `SlowCSS`.

### Lighthouse Flow

```python
# Full audit (accessibility, SEO, best practices)
mcp__chrome__lighthouse_audit()

# Mobile-specific audit
mcp__chrome__lighthouse_audit(device="mobile")

# Snapshot mode (current state, no reload)
mcp__chrome__lighthouse_audit(mode="snapshot")

# Save reports to directory
mcp__chrome__lighthouse_audit(outputDirPath="./lighthouse-reports")
```

Note: Lighthouse audits exclude performance scores. Use Performance Trace for
Core Web Vitals (LCP, INP, CLS).

### Memory Snapshot Flow

```python
# Capture heap snapshot for memory leak analysis
mcp__chrome__take_memory_snapshot(filePath="heap.heapsnapshot")
# Open the .heapsnapshot file in Chrome DevTools for analysis
```

### Emulation Flow

```python
# Mobile viewport
mcp__chrome__emulate(viewport="375x812x3,mobile,touch")

# Tablet landscape
mcp__chrome__emulate(viewport="1024x768x2,touch,landscape")

# Slow network
mcp__chrome__emulate(networkConditions="Slow 3G")

# CPU throttling (4x slowdown)
mcp__chrome__emulate(cpuThrottlingRate=4)

# Dark mode
mcp__chrome__emulate(colorScheme="dark")

# Geolocation (Tokyo)
mcp__chrome__emulate(geolocation="35.6762x139.6503")

# Reset all emulation
mcp__chrome__emulate(
    viewport="1280x720x1",
    colorScheme="auto",
    cpuThrottlingRate=1
)
# Network reset: omit networkConditions parameter
```

### Multi-Page Scraping Flow

```python
# Open multiple pages
mcp__chrome__new_page(url="https://example.com/page1")
mcp__chrome__new_page(url="https://example.com/page2", background=True)

# List all open pages
mcp__chrome__list_pages()

# Switch between pages by ID
mcp__chrome__select_page(pageId=2)
mcp__chrome__take_snapshot()

# Extract data with JavaScript
mcp__chrome__evaluate_script(
    function='() => { return [...document.querySelectorAll(".item")].map(e => e.textContent) }'
)

# Close when done
mcp__chrome__close_page(pageId=2)
```

## Tab Management

Chrome MCP can have multiple pages open. Always be aware of which page is selected.

```python
# See all pages
mcp__chrome__list_pages()

# Select a page (all subsequent commands target this page)
mcp__chrome__select_page(pageId=1, bringToFront=True)

# Open page in isolated context (separate cookies/storage)
mcp__chrome__new_page(url="https://example.com", isolatedContext="session-2")
```

**Rule**: The last open page cannot be closed. Always keep at least one page open.

## Keyboard & Special Interactions

```python
# Press a key
mcp__chrome__press_key(key="Enter")

# Key combinations
mcp__chrome__press_key(key="Control+A")      # Select all
mcp__chrome__press_key(key="Control+C")      # Copy
mcp__chrome__press_key(key="Control+Shift+R") # Hard reload

# Type text into focused input (use after clicking an input)
mcp__chrome__type_text(text="Hello world", submitKey="Enter")

# Handle browser dialogs (alert/confirm/prompt)
mcp__chrome__handle_dialog(action="accept")
mcp__chrome__handle_dialog(action="dismiss")
mcp__chrome__handle_dialog(action="accept", promptText="my input")

# Hover (for tooltips, dropdowns)
mcp__chrome__hover(uid="<element-uid>")

# Drag and drop
mcp__chrome__drag(from_uid="<source-uid>", to_uid="<target-uid>")

# Double click
mcp__chrome__click(uid="<element-uid>", dblClick=True)
```

## Anti-Patterns

❌ **Don't** interact with elements without taking a snapshot first
✅ **Do** always `take_snapshot()` to get current UIDs before clicking/filling

❌ **Don't** reuse UIDs from old snapshots after navigation
✅ **Do** re-take snapshot after every page load or major DOM change

❌ **Don't** use `take_screenshot()` as primary way to understand page content
✅ **Do** use `take_snapshot()` (text-based a11y tree) — it's faster, structured, and gives UIDs

❌ **Don't** forget to check MCP connection status before automation
✅ **Do** verify `mcp(action="status")` shows chrome connected

❌ **Don't** leave tabs open after automation — causes tab leaks
✅ **Do** close pages with `close_page()` when done (keep at least one open)

❌ **Don't** use `navigate_page` when you need a fresh tab with separate state
✅ **Do** use `new_page(isolatedContext="...")` for separate cookie/storage contexts

❌ **Don't** start a performance trace without navigating to the target page first
✅ **Do** navigate first, then `performance_start_trace(reload=True, autoStop=True)`

❌ **Don't** try to use `fill()` for keyboard shortcuts or special keys
✅ **Do** use `press_key()` for Enter, Tab, Escape, keyboard shortcuts

❌ **Don't** wait with arbitrary `sleep()` calls
✅ **Do** use `wait_for(text=["Expected text"])` for deterministic waits

## Reference Files

- [🛡️ Anti-Detection](references/anti-detection.md) — Stealth flags, fingerprint evasion, bot detection bypass
- [🔧 Tool Reference](references/tool-reference.md) — Complete Chrome MCP tool catalog with all parameters
- [📋 Common Recipes](references/common-recipes.md) — Copy-paste patterns for frequent automation tasks
