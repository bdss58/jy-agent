# Language-Specific Review Patterns

## Python

### Common Bugs
```python
# Mutable default argument — shared across all calls!
def append_to(item, target=[]):  # ❌ 
    target.append(item)
    return target
# Fix: use None sentinel
def append_to(item, target=None):  # ✅
    if target is None:
        target = []
    target.append(item)
    return target

# Late binding closure — all lambdas capture i=4
funcs = [lambda: i for i in range(5)]  # ❌ all return 4
funcs = [lambda i=i: i for i in range(5)]  # ✅ each returns 0-4

# String concatenation in loop — O(n²)
result = ""
for s in strings:
    result += s  # ❌ creates new string each time
result = "".join(strings)  # ✅ O(n)
```

### Resource Management
```python
# ❌ Resource leak
f = open("file.txt")
data = f.read()
# f.close() might never be called if an exception occurs

# ✅ Context manager
with open("file.txt") as f:
    data = f.read()

# ✅ Also for: database connections, locks, temp files
with sqlite3.connect("db.sqlite") as conn:
    ...
with tempfile.NamedTemporaryFile() as tmp:
    ...
```

### Async Patterns
```python
# ❌ Running async code in sync context
asyncio.get_event_loop().run_until_complete(coro)  # Deprecated

# ✅ Modern async
asyncio.run(coro)  # Python 3.7+

# ❌ Blocking call in async code
async def handler():
    data = requests.get(url)  # Blocks the event loop!

# ✅ Use async HTTP client
async def handler():
    async with httpx.AsyncClient() as client:
        data = await client.get(url)
```

### Type Hints
```python
# Minimum useful type hints:
def process_users(
    users: list[dict[str, Any]],  # Input types
    limit: int = 100,
) -> list[str]:  # Return type
    ...

# For complex types, use TypedDict or dataclass:
from typing import TypedDict

class User(TypedDict):
    name: str
    email: str
    age: int
```

## JavaScript / TypeScript

### Common Bugs
```javascript
// == vs === confusion
if (value == null)  // matches null AND undefined (sometimes intentional)
if (value === null) // matches only null

// Floating point
0.1 + 0.2 === 0.3  // false! 
Math.abs(0.1 + 0.2 - 0.3) < Number.EPSILON  // ✅

// Array methods that mutate vs return new
const sorted = arr.sort()  // ❌ MUTATES arr in place AND returns it
const sorted = [...arr].sort()  // ✅ Doesn't mutate original

// Async forEach — doesn't await!
arr.forEach(async (item) => {  // ❌ fires all at once, no await
    await process(item)
})
for (const item of arr) {  // ✅ sequential
    await process(item)
}
await Promise.all(arr.map(item => process(item)))  // ✅ parallel
```

### Memory Leaks
```javascript
// Event listeners not cleaned up
element.addEventListener('click', handler)
// Must remove when done:
element.removeEventListener('click', handler)

// Closures holding references
function setup() {
    const hugeData = loadData()  // ❌ hugeData never freed if handler persists
    button.onclick = () => console.log(hugeData.length)
}

// setInterval without cleanup
const id = setInterval(poll, 1000)
// Must: clearInterval(id) when component unmounts
```

### TypeScript-Specific
```typescript
// ❌ `any` defeats the purpose of TypeScript
function process(data: any): any { ... }

// ✅ Use generics or proper types
function process<T>(data: T): ProcessResult<T> { ... }

// ❌ Non-null assertion hides bugs
const value = maybeNull!.property  

// ✅ Proper null handling
const value = maybeNull?.property ?? defaultValue
```

## Bash / Shell

### Common Issues
```bash
# ❌ Unquoted variables — word splitting and globbing
rm $file          # If file="my file.txt", deletes "my" and "file.txt"
rm "$file"        # ✅ Always quote variables

# ❌ Missing error handling
cd /some/dir
rm -rf *          # If cd fails, deletes everything in current dir!

cd /some/dir || exit 1  # ✅ Check cd return value
set -euo pipefail       # ✅ Better: enable strict mode at top of script

# ❌ Using ls for scripting
for f in $(ls *.txt); do  # Breaks on spaces, special chars
for f in *.txt; do        # ✅ Use globbing directly
```
