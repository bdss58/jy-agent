"""
web_fetch — Native tool for fetching URLs and extracting clean readable text.

5-tier anti-blocking cascade:
  1. curl_cffi  — Chrome TLS fingerprint impersonation (fastest, best anti-bot bypass)
  2. httpx      — Standard HTTP with browser-like headers
  3. Jina Reader — JS-rendering proxy (handles SPAs/dynamic content)
  4. Chrome      — Direct Chrome DevTools MCP calls (navigate → extract → close)
  5. Error diagnostics

Supports pagination via start_index/max_length for large pages.

Usage (as native tool):
  web_fetch(url="https://example.com")
  web_fetch(url="https://...", max_length=15000, start_index=5000)
  web_fetch(url="https://...", strategy="jina")
  web_fetch(url="https://...", raw=True)
"""

import re
from urllib.parse import urlparse
from ..toolresult import ToolResult

# ─── Constants ────────────────────────────────────────────────────────────────

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    # NOTE: Only gzip/deflate — NOT br (Brotli).
    # httpx cannot decompress Brotli unless the 'brotli' package is installed.
    # Without it, responses with Content-Encoding: br arrive as garbled binary,
    # which silently passes HTTP status checks but produces unreadable content.
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Separate headers for curl_cffi which handles its own decompression
_CFFI_HEADERS = {
    **_BROWSER_HEADERS,
    "Accept-Encoding": "gzip, deflate, br",  # curl_cffi handles br natively
}

_JINA_PREFIX = "https://r.jina.ai/"

# Minimum content length thresholds
_MIN_CONTENT_LENGTH = 200          # For normal pages
_MIN_CONTENT_LENGTH_SEARCH = 500   # For search result pages (expect more content)

# Regex to strip XML-incompatible control characters (NULL bytes, etc.)
# Keeps: \t (0x09), \n (0x0A), \r (0x0D), and all chars >= 0x20
_XML_ILLEGAL_CHARS_RE = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]'
)

# ─── URL classification ─────────────────────────────────────────────────────

# Domains that are heavily JS-dependent or have strong anti-bot protection.
# For these, simple HTTP fetches (cffi/httpx) almost always fail,
# so we skip straight to Jina/Chrome to save time and avoid false positives.
_JS_HEAVY_DOMAINS = {
    "google.com", "www.google.com", "google.co.jp", "www.google.co.jp",
    "zhihu.com", "www.zhihu.com",
    "twitter.com", "x.com",
    "instagram.com", "www.instagram.com",
    "facebook.com", "www.facebook.com",
    "linkedin.com", "www.linkedin.com",
    "reddit.com", "www.reddit.com", "old.reddit.com",
    "weibo.com", "www.weibo.com", "m.weibo.com",
    "douyin.com", "www.douyin.com",
    "xiaohongshu.com", "www.xiaohongshu.com",
    "taobao.com", "www.taobao.com",
    "jd.com", "www.jd.com",
    "bilibili.com", "www.bilibili.com", "search.bilibili.com",
}


def _is_search_url(url: str) -> bool:
    """Check if a URL is a search results page (expect richer content)."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Google/Bing/Baidu/DuckDuckGo search
    if any(s in host for s in ["google.", "bing.com", "baidu.com", "duckduckgo.com"]):
        if "/search" in path or "q=" in query or "wd=" in query:
            return True

    # Zhihu/Bilibili/Weibo search
    if "search" in path or "search" in host:
        return True

    return False


def _is_js_heavy(url: str) -> bool:
    """Check if a URL belongs to a JS-heavy / anti-bot domain."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    # Check exact match or parent domain match
    if host in _JS_HEAVY_DOMAINS:
        return True
    # Check parent domain (e.g., "search.bilibili.com" → "bilibili.com")
    parts = host.split(".")
    if len(parts) >= 2:
        parent = ".".join(parts[-2:])
        if parent in _JS_HEAVY_DOMAINS or f"www.{parent}" in _JS_HEAVY_DOMAINS:
            return True
    return False


# ─── HTML sanitization ───────────────────────────────────────────────────────

def _sanitize_html(html: str) -> str:
    """Remove NULL bytes and XML-incompatible control characters from HTML.

    This prevents crashes in lxml/readability/trafilatura which cannot handle
    these characters (e.g., Google search results sometimes contain them).
    """
    if not html:
        return html
    return _XML_ILLEGAL_CHARS_RE.sub('', html)


# ─── Garbled content detection ───────────────────────────────────────────────

def _is_garbled(text: str) -> bool:
    """Detect if response text is garbled/binary (e.g. undecoded Brotli).

    When a server sends Brotli-compressed content but the client can't decompress
    it, the raw bytes get decoded as mojibake — high ratio of non-ASCII/replacement
    characters relative to normal text.
    """
    if not text or len(text) < 100:
        return False

    sample = text[:2000]

    # Check 1: High ratio of replacement characters (U+FFFD) or non-printable chars
    non_text_count = sum(1 for c in sample if ord(c) > 0xFFFD or (ord(c) > 127 and ord(c) < 160))
    if non_text_count > len(sample) * 0.3:
        return True

    # Check 2: Very low ratio of ASCII printable characters in what should be HTML
    ascii_printable = sum(1 for c in sample if 32 <= ord(c) <= 126)
    if ascii_printable < len(sample) * 0.3:
        return True

    # Check 3: No common HTML markers in first 500 chars of what should be a web page
    head = sample[:500].lower()
    html_markers = ['<html', '<!doctype', '<head', '<body', '<div', '<meta', '<script', '<link']
    if not any(m in head for m in html_markers):
        # Could be plain text or markdown (from Jina), which is fine
        # Only flag as garbled if also has high non-ASCII ratio
        ascii_ratio = ascii_printable / max(len(sample), 1)
        if ascii_ratio < 0.5:
            return True

    return False


# ─── Low-quality content detection ──────────────────────────────────────────

def _is_low_quality(content: str, url: str = "") -> str | None:
    """Detect fake-success responses that look like they succeeded but have no useful content.

    Returns a reason string if low-quality, None if content looks good.

    Detects:
    - JS redirect pages (Google's "if not redirected, click here")
    - Empty SPA shells (just framework boilerplate, no actual content)
    - Login walls / auth gates
    - Cookie consent walls
    - Generic error pages that return 200
    """
    if not content:
        return "empty content"

    content_stripped = content.strip()
    content_lower = content_stripped.lower()
    content_len = len(content_stripped)

    # ── Pattern 1: JS redirect / meta-refresh pages ──
    # Google returns these when it detects non-browser clients
    redirect_signals = [
        "如果您在几秒钟内没有被重定向",          # Google Chinese redirect
        "if you are not redirected",              # Google English redirect
        "if you're having trouble accessing",     # Google fallback
        "click here if you are not redirected",
        "meta http-equiv=\"refresh\"",
        "window.location.replace",
        "document.location.href",
    ]
    for sig in redirect_signals:
        if sig in content_lower:
            return f"JS/meta redirect page (matched: '{sig[:40]}')"

    # ── Pattern 2: Empty SPA shells ──
    # Pages that loaded the framework but no data (common with SPAs)
    if content_len < 500:
        spa_shells = [
            "id=\"root\"></div>",
            "id=\"app\"></div>",
            "id=\"__next\"></div>",
            "id=\"__nuxt\"></div>",
            "<div id=\"root\">",
            "noscript>you need to enable javascript",
            "noscript>please enable javascript",
        ]
        for sig in spa_shells:
            if sig in content_lower:
                return f"empty SPA shell (matched: '{sig[:40]}')"

    # ── Pattern 3: Minimal stub pages ──
    # Very short content that's technically "text" but useless
    # Count actual word-like tokens (not just whitespace/punctuation)
    words = re.findall(r'\b\w{2,}\b', content_stripped)
    if len(words) < 15 and content_len < 500:
        return f"too few words ({len(words)} words in {content_len} chars)"

    # ── Pattern 4: Login/auth walls ──
    if content_len < 2000:
        auth_signals = [
            "please sign in", "please log in", "登录后查看",
            "请先登录", "sign in to continue", "log in to continue",
            "create an account", "注册账号",
        ]
        # Only match if the page is short (a real page with a login button is fine)
        auth_count = sum(1 for sig in auth_signals if sig in content_lower)
        if auth_count >= 1 and content_len < 500:
            return "login/auth wall"

    # ── Pattern 5: Title-only or URL-source-only pages ──
    # Jina sometimes returns just "Title:\nURL Source:\n" with no body
    lines = [line.strip() for line in content_stripped.split('\n') if line.strip()]
    non_meta_lines = [
        line for line in lines
        if not line.lower().startswith(('title:', 'url source:', 'url:', 'source:'))
    ]
    if len(non_meta_lines) < 3 and content_len < 500:
        return f"metadata-only response ({len(non_meta_lines)} content lines)"

    return None  # Content looks OK


# ─── Text extraction ─────────────────────────────────────────────────────────

def _extract_text(html: str, url: str = "") -> str:
    """Extract clean readable text from HTML. Tries trafilatura → readability+html2text → bs4."""
    if not html or len(html.strip()) < 50:
        return html.strip()

    # Sanitize HTML: strip NULL bytes and control characters that crash parsers
    html = _sanitize_html(html)

    # Strategy 1: trafilatura (best for articles/blog posts)
    try:
        import trafilatura
        result = trafilatura.extract(html, include_links=True, include_tables=True,
                                     include_comments=False, favor_recall=True)
        if result and len(result) > 200:
            return result
    except Exception:
        pass

    # Strategy 2: readability + html2text (good for docs/general pages)
    try:
        from readability import Document
        import html2text
        doc = Document(html)
        clean_html = doc.summary()
        title = doc.title()

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.ignore_emphasis = False
        converter.body_width = 0  # No line wrapping
        converter.skip_internal_links = True

        text = converter.handle(clean_html)
        if title and title not in text[:200]:
            text = f"# {title}\n\n{text}"
        if len(text) > 200:
            return text
    except Exception:
        pass

    # Strategy 3: BeautifulSoup (fallback)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    except Exception:
        pass

    # Last resort: regex strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Fetch strategies ────────────────────────────────────────────────────────

def _fetch_cffi(url: str, timeout: int = 20) -> tuple:
    """Tier 1: curl_cffi with Chrome TLS impersonation."""
    from curl_cffi import requests as cffi_requests
    resp = cffi_requests.get(url, headers=_CFFI_HEADERS, impersonate="chrome",
                             timeout=timeout, allow_redirects=True)
    return resp.status_code, resp.text


def _fetch_httpx(url: str, timeout: int = 20) -> tuple:
    """Tier 2: httpx with browser-like headers."""
    import httpx
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=_BROWSER_HEADERS) as client:
        resp = client.get(url)
        return resp.status_code, resp.text


def _fetch_jina(url: str, timeout: int = 30) -> tuple:
    """Tier 3: Jina Reader proxy (renders JS, returns markdown)."""
    import httpx
    jina_url = f"{_JINA_PREFIX}{url}"
    headers = {"Accept": "text/plain", "User-Agent": _BROWSER_HEADERS["User-Agent"]}
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        resp = client.get(jina_url)
        return resp.status_code, resp.text


# ─── Tier 4: Chrome via direct MCP calls ────────────────────────────────────

# Simple approach: call Chrome MCP tools directly (navigate → extract → close).
# No LLM sub-call needed — just 3 deterministic tool calls.

# JS to extract text content. Used for JS-heavy article/page fetches
# (Twitter, Reddit, Zhihu, Weibo, etc.) as a last-resort fallback.
#
# NOTE: Chrome is NOT used for search-engine SERP scraping any more —
# the `web_search` tool provides a dedicated multi-engine cascade
# (DDG / Brave / Mojeek / SearxNG) that's far cheaper and more reliable.
_CHROME_EXTRACT_JS = """() => {
    return document.body.innerText;
}"""


def _fetch_chrome(url: str, timeout: int = 30) -> tuple:
    """Tier 4: Fetch via Chrome DevTools MCP — direct tool calls, no LLM.

    Delegates to MCPManager.chrome_fetch_page() which handles:
    - Reference-counted Chrome connection (no premature disconnect)
    - Page-level locking (no interleaving between concurrent callers)
    - Explicit select_page before evaluate_script / take_snapshot
    - Automatic cleanup (close tab, restore original selection)

    Used as a last-resort fallback for JS-heavy article pages only.
    SERP scraping belongs in `web_search`, not here.

    Raises RuntimeError on failure (causes fallthrough to next strategy).
    """
    from ..mcp_manager import get_manager

    manager = get_manager()
    content = manager.chrome_fetch_page(
        url, timeout=timeout, js_function=_CHROME_EXTRACT_JS,
    )
    return 200, content


# ─── Strategy orchestration ──────────────────────────────────────────────────

_STRATEGY_MAP = {
    "cffi": [_fetch_cffi],
    "direct": [_fetch_httpx],
    "jina": [_fetch_jina],
    "chrome": [_fetch_chrome],
    "auto": [_fetch_cffi, _fetch_httpx, _fetch_jina, _fetch_chrome],
}

# For JS-heavy/anti-bot sites (Twitter/X, Reddit, Zhihu, Weibo, etc.), skip
# simple HTTP and go straight to Jina/Chrome. Note: search engine SERPs are
# now handled exclusively by the `web_search` tool, not by web_fetch+Chrome.
_STRATEGY_MAP_JS_HEAVY = {
    "auto": [_fetch_jina, _fetch_cffi, _fetch_chrome],
}

# Jina and Chrome snapshot both return pre-extracted text — skip HTML extraction
_RETURNS_TEXT = {_fetch_jina, _fetch_chrome}


def _is_blocked(status: int, body: str) -> bool:
    """Detect if a response is a block/captcha page rather than real content."""
    if status in (403, 429, 503, 520, 521, 522, 523, 524):
        return True
    if status == 200 and len(body) < 2000:
        block_signals = [
            "captcha", "challenge", "cf-browser-verification",
            "access denied", "blocked", "please verify",
            "enable javascript", "just a moment",
            "checking your browser", "ray id",
            "attention required", "cloudflare",
            "sorry, you have been blocked",
            "unusual traffic from your computer",
        ]
        body_lower = body.lower()
        if any(sig in body_lower for sig in block_signals):
            return True
    return False


# ─── Main function ────────────────────────────────────────────────────────────

def web_fetch(url: str, max_length: int = 8000, start_index: int = 0,
              raw: bool = False, strategy: str = "auto") -> ToolResult:
    """Fetch a URL and return its content as clean readable text.

    5-tier anti-blocking cascade: curl_cffi (Chrome TLS impersonation) →
    httpx (browser headers) → Jina Reader (JS rendering proxy) →
    Chrome (agent's interactive browser) → error diagnostics.

    Smart URL detection: JS-heavy sites (Google, Zhihu, Twitter, etc.) skip
    straight to Jina/Chrome to avoid wasting time on doomed HTTP fetches.

    Supports pagination via start_index/max_length for large pages.

    Args:
        url: URL to fetch
        max_length: Maximum characters to return per page (default 8000)
        start_index: Start position for pagination (default 0)
        raw: If True, return raw HTML without text extraction (default False)
        strategy: Fetch strategy — auto, cffi, direct, jina, chrome (default "auto")

    Returns:
        Formatted string with URL, status, content length, and extracted text.
    """
    if not url:
        return ToolResult("Error: url parameter is required", is_error=True)

    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Smart strategy selection for JS-heavy sites
    is_search = _is_search_url(url)
    js_heavy = _is_js_heavy(url)

    if strategy == "auto" and js_heavy:
        fetchers = _STRATEGY_MAP_JS_HEAVY["auto"]
    else:
        fetchers = _STRATEGY_MAP.get(strategy, _STRATEGY_MAP["auto"])

    # Determine minimum content threshold
    min_length = _MIN_CONTENT_LENGTH_SEARCH if is_search else _MIN_CONTENT_LENGTH

    errors = []

    for fetcher in fetchers:
        strategy_name = fetcher.__name__.replace("_fetch_", "")
        try:
            status, body = fetcher(url)

            # Check for garbled/binary content (e.g. undecoded Brotli)
            if _is_garbled(body):
                errors.append(f"{strategy_name}: garbled/binary content (possible encoding issue)")
                continue

            if _is_blocked(status, body):
                errors.append(f"{strategy_name}: blocked (HTTP {status})")
                continue

            if status >= 400:
                errors.append(f"{strategy_name}: HTTP {status}")
                # Short-circuit: 404/410 means resource doesn't exist — no point
                # trying other strategies on the same non-existent URL.
                if status in (404, 410) and strategy == "auto":
                    error_detail = "\n".join(f"  • {e}" for e in errors)
                    return ToolResult(
                        f"Error: HTTP {status} — Resource not found: {url}\n"
                        f"(Short-circuited after {strategy_name} — retrying won't help)\n\n"
                        f"Details:\n{error_detail}",
                        is_error=True)
                continue

            # Extract text (unless raw mode or fetcher returns pre-extracted text)
            if raw:
                content = body
            elif fetcher in _RETURNS_TEXT:
                content = body
            else:
                content = _extract_text(body, url)

            # ── Content quality checks ──

            # Check 1: Minimum length
            content_stripped = content.strip() if content else ""
            if len(content_stripped) < min_length:
                errors.append(f"{strategy_name}: too-short content ({len(content_stripped)} chars, need {min_length})")
                continue

            # Check 2: Low-quality / fake-success detection
            quality_issue = _is_low_quality(content_stripped, url)
            if quality_issue:
                errors.append(f"{strategy_name}: low-quality content — {quality_issue}")
                continue

            # Content passed all quality checks!

            # Apply pagination
            total_length = len(content)
            page = content[start_index:start_index + max_length]
            remaining = total_length - start_index - len(page)

            # Build response header
            header = f"URL: {url}\nStatus: {status} | Strategy: {strategy_name}\nContent Length: {total_length} chars"
            if start_index > 0 or remaining > 0:
                header += f"\nShowing: {start_index}-{start_index + len(page)}"
                if remaining > 0:
                    header += f" ({remaining} chars remaining, use start_index={start_index + len(page)} for next page)"

            return ToolResult(f"{header}\n\n{page}")

        except Exception as e:
            errors.append(f"{strategy_name}: {type(e).__name__}: {e}")
            continue

    # All strategies failed
    error_detail = "\n".join(f"  • {e}" for e in errors)
    return ToolResult(f"Error: All fetch strategies failed for {url}\n\nDetails:\n{error_detail}", is_error=True)


# ─── Tool schema for auto-discovery by tools.py ──────────────────────────────

TOOL_SCHEMA = {
    "name": "web_fetch",
    "description": "Fetch a URL and return its content as clean readable text. 5-tier anti-blocking cascade: curl_cffi (Chrome TLS impersonation) → httpx (browser headers) → Jina Reader (JS rendering proxy) → Chrome (agent's interactive browser) → error diagnostics. Supports pagination via start_index/max_length.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch"
            },
            "max_length": {
                "type": "integer",
                "description": "Max chars per page (default 8000)"
            },
            "start_index": {
                "type": "integer",
                "description": "Start index for pagination (default 0)"
            },
            "raw": {
                "type": "boolean",
                "description": "Return raw content without text extraction (default false)"
            },
            "strategy": {
                "type": "string",
                "description": "Fetch strategy: auto, cffi, direct, jina, chrome (default auto)"
            }
        },
        "required": ["url"]
    }
}
