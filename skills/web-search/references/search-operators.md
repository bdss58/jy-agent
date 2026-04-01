# Search Engine Operators — Quick Reference

## Google

### Basic Operators
| Operator | Example | Purpose |
|----------|---------|---------|
| `"exact phrase"` | `"connection refused" python` | Match exact string |
| `site:` | `site:github.com fastapi` | Restrict to a domain |
| `filetype:` | `filetype:pdf k8s security` | Find specific file types |
| `-keyword` | `python async -asyncio` | Exclude a term |
| `OR` | `FastAPI OR Django` | Either term |
| `intitle:` | `intitle:benchmark llama3` | Term must be in page title |
| `inurl:` | `inurl:api/v2 docs` | Term must be in URL |
| `related:` | `related:fastapi.tiangolo.com` | Find similar sites |
| `cache:` | `cache:example.com/page` | Google's cached version |

### Date Filters
| Operator | Example | Purpose |
|----------|---------|---------|
| `after:YYYY-MM-DD` | `after:2024-01-01 LLM agents` | Results after date |
| `before:YYYY-MM-DD` | `before:2024-06-01 react 18` | Results before date |
| `tbs=qdr:h` | URL param | Past hour |
| `tbs=qdr:d` | URL param | Past 24 hours |
| `tbs=qdr:w` | URL param | Past week |
| `tbs=qdr:m` | URL param | Past month |
| `tbs=qdr:y` | URL param | Past year |

### Special Search Types
| URL Param | Purpose |
|-----------|---------|
| `tbm=nws` | Google News |
| `tbm=isch` | Google Images |
| `tbm=vid` | Google Videos |

### Power Combos
```
# Find recent GitHub issues about a specific error
site:github.com "TypeError: Cannot read property" after:2024-06-01

# Find PDF whitepapers on a topic
filetype:pdf "large language model" "fine-tuning" after:2024-01-01

# Find alternatives to a tool
"alternative to" terraform infrastructure-as-code -advertisement

# Find official documentation for a specific API
site:docs.aws.amazon.com "lambda" "container image"
```

## DuckDuckGo

### Operators (subset of Google's)
| Operator | Example | Purpose |
|----------|---------|---------|
| `"exact phrase"` | `"exact match"` | Exact string |
| `site:` | `site:stackoverflow.com` | Restrict domain |
| `filetype:` | `filetype:pdf` | File type |
| `-keyword` | `-pinterest` | Exclude |
| `intitle:` | `intitle:tutorial` | In title |

### Bang Shortcuts (redirect to other engines)
| Bang | Redirects to |
|------|-------------|
| `!g` | Google |
| `!gh` | GitHub |
| `!so` | Stack Overflow |
| `!w` | Wikipedia |
| `!mdn` | MDN Web Docs |
| `!py` | Python docs |
| `!npm` | npm registry |

**Usage:** `web_fetch("https://duckduckgo.com/html/?q=!gh+fastapi+middleware")`

## 百度

### Operators
| Operator | Example | Purpose |
|----------|---------|---------|
| `"精确匹配"` | `"连接超时" python` | Exact match |
| `site:` | `site:zhihu.com LLM` | Restrict domain |
| `filetype:` | `filetype:pdf 机器学习` | File type |
| `-关键词` | `python教程 -广告` | Exclude |
| `intitle:` | `intitle:教程 kubernetes` | In title |
| `inurl:` | `inurl:blog python` | In URL |

### 百度 Time Filters (URL params)
```python
# Past day
web_fetch("https://www.baidu.com/s?wd=query&gpc=stf%3D1d")
# Past week
web_fetch("https://www.baidu.com/s?wd=query&gpc=stf%3D1w")
# Past month
web_fetch("https://www.baidu.com/s?wd=query&gpc=stf%3D1m")
# Past year
web_fetch("https://www.baidu.com/s?wd=query&gpc=stf%3D1y")
```
