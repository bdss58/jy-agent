# web_fetch

- 5-tier cascade: `curl_cffi` → `httpx` → Jina Reader → Chrome MCP → error diagnostics
- `curl_cffi` uses Chrome impersonation; `httpx` avoids Brotli unless support is installed, so garbled-response detection catches undecoded content
- HTML is sanitized with `_sanitize_html()` to strip NULL/control chars before `trafilatura` / `readability` / BeautifulSoup
- JS-heavy / anti-bot domains (Google, Zhihu, X, Bilibili, etc.) skip doomed simple HTTP attempts and go straight to Jina/Chrome
- JS-heavy search pages prefer Chrome first because the search-specific JS extractor captures real `<a href>` URLs; Jina / plain innerText often only expose display URLs
- Chrome tier is deterministic direct MCP calls, not an LLM loop: open tab → `evaluate_script` → optional `take_snapshot` fallback → close tab
- Chrome tier auto-connects if needed and disconnects again if it opened Chrome itself
- Quality gates reject fake-success pages: JS/meta redirects, empty SPA shells, metadata-only Jina pages, short stubs, login walls
- Block detection treats 403/429/503-family responses and short captcha/challenge pages as blocked even if HTTP status is 200
- Jina and Chrome return pre-extracted text, so HTML extraction is skipped for those strategies
