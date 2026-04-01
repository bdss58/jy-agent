# Security Review Checklist

Review each category for the language/framework in use.

## Injection Attacks

### SQL Injection
```python
# ❌ VULNERABLE — string formatting in queries
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
cursor.execute("SELECT * FROM users WHERE id = " + user_id)

# ✅ SAFE — parameterized queries
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

### Command Injection
```python
# ❌ VULNERABLE — user input in shell commands
os.system(f"convert {filename} output.png")
subprocess.run(f"grep {pattern} {filepath}", shell=True)

# ✅ SAFE — use lists, avoid shell=True
subprocess.run(["convert", filename, "output.png"])
subprocess.run(["grep", pattern, filepath])
```

### Path Traversal
```python
# ❌ VULNERABLE — user controls file path
filepath = os.path.join(BASE_DIR, user_input)
open(filepath)  # user_input = "../../etc/passwd"

# ✅ SAFE — resolve and verify
filepath = os.path.realpath(os.path.join(BASE_DIR, user_input))
if not filepath.startswith(os.path.realpath(BASE_DIR)):
    raise ValueError("Path traversal detected")
```

### Template Injection (XSS / SSTI)
```python
# ❌ VULNERABLE — raw user input in HTML
return f"<h1>Hello {username}</h1>"

# ✅ SAFE — use template engine with auto-escaping
return render_template("hello.html", username=username)
```

## Authentication & Authorization

- [ ] Passwords hashed with bcrypt/scrypt/argon2 (NOT md5/sha1/sha256)
- [ ] Timing-safe comparison for tokens (`hmac.compare_digest`)
- [ ] Session tokens have sufficient entropy (≥128 bits)
- [ ] Authorization checked on every endpoint (not just the frontend)
- [ ] JWT tokens validated (signature, expiry, issuer)
- [ ] Rate limiting on auth endpoints

## Sensitive Data

- [ ] No secrets in source code (API keys, passwords, tokens)
- [ ] No secrets in logs (mask PII, tokens, passwords)
- [ ] Environment variables or secret managers for credentials
- [ ] HTTPS enforced for sensitive data in transit
- [ ] Sensitive data encrypted at rest where required

## Dependencies

- [ ] No known CVEs in dependencies (`pip audit`, `npm audit`)
- [ ] Pinned versions (no `*` or `latest`)
- [ ] Minimal dependency surface (don't import a library for one function)

## Python-Specific

- [ ] No `eval()` or `exec()` on user input
- [ ] No `pickle.loads()` on untrusted data (use JSON instead)
- [ ] No `yaml.load()` without `Loader=SafeLoader`
- [ ] `subprocess` calls use lists, not strings with `shell=True`
- [ ] `os.path.join` + traversal check for file operations

## JavaScript/TypeScript-Specific

- [ ] No `eval()`, `new Function()`, or `innerHTML` with user data
- [ ] Content-Security-Policy headers set
- [ ] `JSON.parse()` wrapped in try/catch
- [ ] No `dangerouslySetInnerHTML` without sanitization (React)
- [ ] CORS configured restrictively (not `*`)

## API-Specific

- [ ] Input validation on all endpoints (size, type, range)
- [ ] Rate limiting implemented
- [ ] Error responses don't leak internal details (stack traces, SQL)
- [ ] CORS, CSP, X-Frame-Options headers configured
- [ ] File upload limits and type validation
