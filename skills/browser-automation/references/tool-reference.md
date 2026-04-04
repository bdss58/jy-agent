# Chrome MCP Tool Reference

Complete catalog of all Chrome DevTools MCP tools with parameters and usage notes.

## Input Automation

### click
Click an element on the page.
- `uid` (required): Element UID from snapshot
- `dblClick` (optional): Set `true` for double-click
- `includeSnapshot` (optional): Return snapshot after click

### drag
Drag one element onto another.
- `from_uid` (required): Source element UID
- `to_uid` (required): Target element UID
- `includeSnapshot` (optional): Return snapshot after drag

### fill
Type text into an input, textarea, or select an option from `<select>`.
- `uid` (required): Element UID
- `value` (required): Text to enter or option to select
- `includeSnapshot` (optional): Return snapshot after fill

### fill_form
Fill multiple form elements at once (more efficient than individual `fill` calls).
- `elements` (required): Array of `{"uid": "...", "value": "..."}`
- `includeSnapshot` (optional): Return snapshot after fill

### handle_dialog
Respond to browser dialogs (alert, confirm, prompt).
- `action` (required): `"accept"` or `"dismiss"`
- `promptText` (optional): Text to enter in prompt dialogs

### hover
Hover over an element (triggers tooltips, dropdowns, etc.).
- `uid` (required): Element UID
- `includeSnapshot` (optional): Return snapshot after hover

### press_key
Press a key or key combination. Use for Enter, Tab, Escape, shortcuts.
- `key` (required): Key name or combo, e.g., `"Enter"`, `"Control+A"`, `"Control+Shift+R"`
- `includeSnapshot` (optional): Return snapshot after keypress

### type_text
Type text character-by-character into a focused input (more human-like than `fill`).
- `text` (required): Text to type
- `submitKey` (optional): Key to press after typing, e.g., `"Enter"`, `"Tab"`

### upload_file
Upload a file through a file input element.
- `uid` (required): File input element UID
- `filePath` (required): Local path to file
- `includeSnapshot` (optional): Return snapshot after upload

## Navigation

### navigate_page
Navigate current page by URL, or go back/forward/reload.
- `url` (optional): Target URL (only for `type="url"`)
- `type` (optional): `"url"` (default), `"back"`, `"forward"`, `"reload"`
- `initScript` (optional): JS to run before any site scripts on next navigation
- `timeout` (optional): Max wait in ms
- `ignoreCache` (optional): Ignore cache on reload
- `handleBeforeUnload` (optional): `"accept"` (default) or `"decline"`

### new_page
Open a new tab and load a URL.
- `url` (required): URL to load
- `background` (optional): Open in background without focus
- `isolatedContext` (optional): Named browser context (separate cookies/storage)
- `timeout` (optional): Max wait in ms

### select_page
Switch active page for subsequent commands.
- `pageId` (required): Page ID (from `list_pages`)
- `bringToFront` (optional): Focus and raise the page

### list_pages
List all open pages. No parameters.

### close_page
Close a page by its ID. The last open page cannot be closed.
- `pageId` (required): Page ID to close

### wait_for
Wait for text to appear on the page (deterministic waiting).
- `text` (required): Array of strings — resolves when ANY appears
- `timeout` (optional): Max wait in ms

## Emulation

### emulate
Configure device emulation. All parameters are optional — omit to keep current.
- `viewport`: `"<width>x<height>x<dpr>[,mobile][,touch][,landscape]"`
  - Examples: `"375x812x3,mobile,touch"`, `"1024x768x2,touch,landscape"`
- `networkConditions`: `"Offline"`, `"Slow 3G"`, `"Fast 3G"`, `"Slow 4G"`, `"Fast 4G"` (omit to disable)
- `cpuThrottlingRate`: 1-20 (1 = no throttling)
- `colorScheme`: `"dark"`, `"light"`, `"auto"` (reset)
- `geolocation`: `"<lat>x<lon>"` (omit to clear)
- `userAgent`: Custom UA string (empty string to clear)

### resize_page
Resize the page window.
- `width` (required): Page width in pixels
- `height` (required): Page height in pixels

## Performance Analysis

### performance_start_trace
Start recording a performance trace.
- `reload` (optional): Reload page after trace starts (recommended for page-load analysis)
- `autoStop` (optional): Automatically stop after page load
- `filePath` (optional): Save raw trace to file (`.json` or `.json.gz`)

### performance_stop_trace
Stop an active trace recording.
- `filePath` (optional): Save raw trace to file

### performance_analyze_insight
Get detailed analysis of a specific performance insight.
- `insightSetId` (required): ID from trace results
- `insightName` (required): Insight name, e.g.:
  - `"LCPBreakdown"` — Largest Contentful Paint breakdown
  - `"CLSCulprits"` — Cumulative Layout Shift sources
  - `"DocumentLatency"` — Document load timing
  - `"RenderBlocking"` — Render-blocking resources
  - `"ThirdParties"` — Third-party script impact
  - `"SlowCSS"` — Slow CSS selectors
  - `"InteractionToNextPaint"` — INP details

## Network Debugging

### list_network_requests
List all requests since last navigation.
- `resourceTypes` (optional): Filter by type array: `"document"`, `"stylesheet"`, `"image"`, `"media"`, `"font"`, `"script"`, `"xhr"`, `"fetch"`, `"websocket"`, etc.
- `pageIdx` (optional): Page number (0-based) for pagination
- `pageSize` (optional): Max requests per page
- `includePreservedRequests` (optional): Include requests from last 3 navigations

### get_network_request
Get details of a specific request.
- `reqid` (optional): Request ID (omit for currently selected in DevTools)
- `requestFilePath` (optional): Save request body to file
- `responseFilePath` (optional): Save response body to file

## Browser Debugging

### evaluate_script
Execute JavaScript in the page context. Returns JSON-serializable results.
- `function` (required): JS function string, e.g., `'() => { return document.title }'`
- `args` (optional): Array of element UIDs passed as arguments

### list_console_messages
List console messages since last navigation.
- `types` (optional): Filter array: `"log"`, `"error"`, `"warn"`, `"info"`, `"debug"`, etc.
- `pageIdx` (optional): Page number (0-based)
- `pageSize` (optional): Max messages per page
- `includePreservedMessages` (optional): Include messages from last 3 navigations

### get_console_message
Get details of a specific console message.
- `msgid` (required): Message ID from `list_console_messages`

### take_snapshot
Take a text snapshot based on the accessibility tree. **Preferred over screenshots.**
- `verbose` (optional): Include all a11y properties
- `filePath` (optional): Save to file instead of inline

### take_screenshot
Capture a visual screenshot.
- `fullPage` (optional): Full page instead of viewport (incompatible with `uid`)
- `uid` (optional): Screenshot a specific element
- `filePath` (optional): Save to file
- `format` (optional): `"png"` (default), `"jpeg"`, `"webp"`
- `quality` (optional): 0-100 for JPEG/WebP compression

### take_memory_snapshot
Capture a V8 heap snapshot for memory analysis.
- `filePath` (required): Path to save `.heapsnapshot` file

## Audits

### lighthouse_audit
Run Lighthouse audit for accessibility, SEO, and best practices.
**Note**: Excludes performance — use performance traces for Core Web Vitals.
- `device` (optional): `"desktop"` or `"mobile"`
- `mode` (optional): `"navigation"` (reloads page) or `"snapshot"` (current state)
- `outputDirPath` (optional): Directory for HTML/JSON reports
