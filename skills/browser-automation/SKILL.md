---
name: browser-automation
description: >-
  Automate Chrome browser interactions using MCP Chrome DevTools. Use when asked 
  to navigate websites, fill forms, click buttons, take screenshots, scrape dynamic 
  content, debug web pages, or interact with web applications like Taobao, Zhihu, etc.
metadata:
  author: agent-builtin
  version: "1.0"
---

## Instructions

When automating browser interactions via Chrome MCP:

### 1. Connection Setup
- Ensure Chrome MCP is connected: use `mcp` tool with action `connect`, server `chrome`
- If connection fails, Chrome may need to be running with remote debugging enabled

### 2. Navigation Pattern
Always follow this sequence:
1. **Navigate**: `navigate_page` to the target URL
2. **Snapshot**: `take_snapshot` to see the page structure (a11y tree with UIDs)
3. **Act**: Use UIDs from the snapshot to click, fill, type
4. **Verify**: Take another snapshot or screenshot to confirm the result

### 3. Element Interaction
- **ALWAYS take a snapshot first** before trying to interact with elements
- Use the `uid` from the snapshot to reference elements
- For text inputs: use `fill` (sets value) or `type_text` (simulates typing)
- For buttons/links: use `click`
- For dropdowns/selects: use `fill` with the option value
- For keyboard shortcuts: use `press_key` (e.g., "Enter", "Control+A")

### 4. Handling Dynamic Content
- After clicking, wait for content to load: use `wait_for` with expected text
- For SPAs (Single Page Apps), take a new snapshot after navigation
- Use `evaluate_script` for complex DOM operations

### 5. Screenshots & Debugging
- Use `take_screenshot` for visual verification
- Use `take_snapshot` for structural analysis (preferred — uses less tokens)
- Use `list_console_messages` to check for JavaScript errors
- Use `list_network_requests` to debug API calls

### 6. Multi-Page Workflows
- Use `list_pages` to see all open tabs
- Use `select_page` to switch between tabs  
- Use `new_page` to open a new tab
- Use `close_page` to close tabs when done

### 7. Common Patterns

#### Login to a website:
```
1. navigate_page → URL
2. take_snapshot → find login form UIDs
3. fill → username field
4. fill → password field  
5. click → submit button
6. wait_for → dashboard text
7. take_snapshot → verify logged in
```

#### Scrape dynamic content:
```
1. navigate_page → target URL
2. wait_for → key content text
3. take_snapshot → get element structure
4. evaluate_script → extract specific data via JS
```

#### Fill a complex form:
```
1. take_snapshot → identify all form fields
2. fill_form → fill multiple fields at once
3. take_screenshot → visual verification
4. click → submit
```

### 8. Anti-Detection Tips
- Don't navigate too rapidly between pages
- Use realistic viewport sizes: `resize_page` width=1440, height=900
- For sites that detect automation, add delays between actions
