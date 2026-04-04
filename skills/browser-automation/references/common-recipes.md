# Common Recipes

Copy-paste patterns for frequent browser automation tasks.

## Login Flow

```python
# Navigate to login page
mcp__chrome__navigate_page(url="https://example.com/login")
mcp__chrome__take_snapshot()

# Fill credentials and submit
mcp__chrome__fill_form(elements=[
    {"uid": "<username-uid>", "value": "user@example.com"},
    {"uid": "<password-uid>", "value": "secret123"}
])
mcp__chrome__click(uid="<login-button-uid>")

# Wait for successful redirect
mcp__chrome__wait_for(text=["Dashboard", "Welcome"])
mcp__chrome__take_snapshot()
```

## Cookie Banner Dismissal

```python
mcp__chrome__take_snapshot()
# Look for "Accept", "Accept All", "OK", "Got it" buttons in snapshot
mcp__chrome__click(uid="<accept-cookies-uid>")
```

## Scroll to Bottom (Infinite Scroll)

```python
# Scroll page to bottom
mcp__chrome__evaluate_script(
    function='() => { window.scrollTo(0, document.body.scrollHeight) }'
)

# Wait for new content to load
mcp__chrome__wait_for(text=["Load more", "some expected content"])

# Or scroll incrementally
mcp__chrome__evaluate_script(
    function='() => { window.scrollBy(0, 800) }'
)
```

## Extract Table Data

```python
mcp__chrome__evaluate_script(
    function='''() => {
        const rows = [...document.querySelectorAll("table tbody tr")];
        return rows.map(row => {
            const cells = [...row.querySelectorAll("td")];
            return cells.map(c => c.textContent.trim());
        });
    }'''
)
```

## Extract All Links

```python
mcp__chrome__evaluate_script(
    function='''() => {
        return [...document.querySelectorAll("a[href]")].map(a => ({
            text: a.textContent.trim(),
            href: a.href
        }));
    }'''
)
```

## Wait for Element to Appear

```python
# Wait for specific text
mcp__chrome__wait_for(text=["Expected Text"])

# Wait for element via JS (with timeout)
mcp__chrome__evaluate_script(
    function='''() => {
        return new Promise((resolve, reject) => {
            const check = () => {
                const el = document.querySelector(".my-element");
                if (el) return resolve(true);
                setTimeout(check, 200);
            };
            check();
            setTimeout(() => reject("Timeout waiting for element"), 10000);
        });
    }'''
)
```

## Screenshot Comparison Workflow

```python
# Take "before" screenshot
mcp__chrome__take_screenshot(filePath="before.png")

# Perform some action
mcp__chrome__click(uid="<action-uid>")

# Take "after" screenshot
mcp__chrome__take_screenshot(filePath="after.png")

# Compare manually or with imagemagick:
# run_shell("compare before.png after.png diff.png")
```

## Download a File

```python
# Click download link — file saves to Chrome's default download dir
mcp__chrome__click(uid="<download-link-uid>")

# Or trigger download via JS
mcp__chrome__evaluate_script(
    function='''() => {
        const a = document.createElement("a");
        a.href = "/path/to/file.pdf";
        a.download = "file.pdf";
        a.click();
    }'''
)
```

## Handle Dropdown / Select

```python
# For native <select> elements, use fill() with the option text
mcp__chrome__fill(uid="<select-uid>", value="Option Text")

# For custom dropdowns (div-based):
# 1. Click to open
mcp__chrome__click(uid="<dropdown-trigger-uid>")
mcp__chrome__take_snapshot()  # Re-snapshot to see dropdown options
# 2. Click the option
mcp__chrome__click(uid="<option-uid>")
```

## Check Page Title / URL

```python
# Get current page title
mcp__chrome__evaluate_script(
    function='() => { return document.title }'
)

# Get current URL
mcp__chrome__evaluate_script(
    function='() => { return window.location.href }'
)
```

## Multi-Tab Workflow

```python
# Open a second page in background
mcp__chrome__new_page(url="https://example.com/page2", background=True)

# List pages to see all tabs
mcp__chrome__list_pages()

# Work on first page
mcp__chrome__select_page(pageId=1)
mcp__chrome__take_snapshot()

# Switch to second page
mcp__chrome__select_page(pageId=2)
mcp__chrome__take_snapshot()

# Close second page when done
mcp__chrome__close_page(pageId=2)
```

## Isolated Sessions (Separate Cookies)

```python
# Open two pages with different login sessions
mcp__chrome__new_page(url="https://app.com/login", isolatedContext="user-A")
# ... login as user A ...

mcp__chrome__new_page(url="https://app.com/login", isolatedContext="user-B")
# ... login as user B ...

# Each context has its own cookies — great for testing multi-user flows
```

## Mobile Responsive Testing

```python
# Emulate iPhone 14 Pro
mcp__chrome__emulate(viewport="393x852x3,mobile,touch")
mcp__chrome__navigate_page(url="https://example.com")
mcp__chrome__take_screenshot(filePath="mobile.png")

# Switch to tablet
mcp__chrome__emulate(viewport="768x1024x2,touch")
mcp__chrome__take_screenshot(filePath="tablet.png")

# Back to desktop
mcp__chrome__emulate(viewport="1440x900x2")
mcp__chrome__take_screenshot(filePath="desktop.png")
```

## Full Page Audit (Quick)

```python
# Navigate to page
mcp__chrome__navigate_page(url="https://example.com")

# 1. Visual check
mcp__chrome__take_screenshot(fullPage=True, filePath="fullpage.png")

# 2. Console errors
mcp__chrome__list_console_messages(types=["error"])

# 3. Network failures
mcp__chrome__list_network_requests(resourceTypes=["xhr", "fetch"])

# 4. Lighthouse audit
mcp__chrome__lighthouse_audit(outputDirPath="./audit-reports")

# 5. Performance trace
mcp__chrome__performance_start_trace(reload=True, autoStop=True)
```

## Pre-Load Script (Anti-Detection + Custom Setup)

```python
mcp__chrome__navigate_page(
    url="https://protected-site.com",
    initScript='''
        // Hide automation
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.chrome = window.chrome || {};
        window.chrome.runtime = window.chrome.runtime || {};
        
        // Custom setup
        console.log("Init script loaded");
    '''
)
```

## Get Page Load Metrics

```python
mcp__chrome__evaluate_script(
    function='''() => {
        const timing = performance.getEntriesByType("navigation")[0];
        return {
            dns: timing.domainLookupEnd - timing.domainLookupStart,
            tcp: timing.connectEnd - timing.connectStart,
            ttfb: timing.responseStart - timing.requestStart,
            domContentLoaded: timing.domContentLoadedEventEnd - timing.fetchStart,
            load: timing.loadEventEnd - timing.fetchStart,
            domInteractive: timing.domInteractive - timing.fetchStart
        };
    }'''
)
```
