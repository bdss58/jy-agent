---
name: web-research
description: >-
  Research topics on the web using multi-strategy fetching. Use when asked to 
  look up information, find documentation, check current events, compare products, 
  or gather data from websites. Includes strategies for anti-blocking and JS-rendered pages.
metadata:
  author: agent-builtin
  version: "1.0"
---

## Instructions

When conducting web research:

### 1. Plan the Research
- Break the topic into specific, searchable queries
- Identify the most authoritative sources for the topic
- Consider multiple perspectives and sources

### 2. Fetch Strategy Selection
Use `web_fetch` with the appropriate strategy:
- **auto** (default): Tries cffi → httpx → Jina → Chrome cascade
- **cffi**: Best for most sites, uses Chrome TLS fingerprint
- **jina**: For JS-heavy sites that need rendering (prefix URL with `https://r.jina.ai/`)
- **chrome**: For sites requiring real browser (login walls, complex JS)

### 3. Search Techniques
- **Google 优先**：使用 `web_fetch` 搜索 `https://www.google.com/search?q=your+query`
  - Google 会被 cffi/httpx 拦截，但 auto 模式的 Chrome tier 可以成功通过
  - 如果 Chrome 不可用，回退到 DuckDuckGo HTML 版：`https://duckduckgo.com/html/?q=your+query`
- 中文搜索结合中英文源（百度、新华网、BBC、Reuters 等）
- For technical docs, go directly to official documentation sites
- For code examples, check GitHub directly
- Use `start_index` and `max_length` for paginating long content

### 4. Quality Standards
- **Always cite sources**: Include URLs where information was found
- **Cross-reference**: Don't rely on a single source for important claims
- **Check dates**: Note when information was published/updated
- **Be transparent**: If a fetch fails or content is unclear, say so
- **Summarize effectively**: Extract key points, don't dump raw HTML

### 5. Anti-Blocking Tips
- If a site blocks, try a different strategy (cffi → jina → chrome)
- Use `max_length=8000` to avoid excessive content
- Space out requests to the same domain
- For paywalled content, look for cached/archive versions

### 6. Common Patterns

#### Research a topic:
```
1. Search Google for the topic
2. Fetch top 3-5 results  
3. Cross-reference findings
4. Synthesize into a clear summary with citations
```

#### Check documentation:
```
1. Go directly to the official docs URL
2. Use pagination (start_index) for long pages
3. Extract relevant sections
```

#### Compare options:
```
1. Research each option independently
2. Create a comparison table
3. Note pros/cons with evidence
```
