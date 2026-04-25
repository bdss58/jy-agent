# SearxNG (private metasearch)

Backing service for jy-agent's `web_search` tool.

`web_search` cascades **SearxNG → DuckDuckGo → Brave → Mojeek**; SearxNG
jumps to first when `SEARXNG_URL` is set in the shell env. With this stack
running locally, set:

```bash
export SEARXNG_URL=http://localhost:18888
```

## Run

```bash
docker compose up -d        # start
docker compose ps           # status (look for "(healthy)")
docker compose logs -f      # follow logs
docker compose down         # stop (volumes survive — they're external)
docker compose pull && docker compose up -d   # upgrade
```

## Editing config

`settings.yml` in this repo is a **slim override file** — it sets only the
keys jy-agent needs to differ from upstream defaults, and uses
[`use_default_settings: true`](https://docs.searxng.org/admin/settings/settings.html#use-default-settings)
to inherit everything else (281 engines, UI, network, etc.) from the default
config baked into the searxng/searxng image.

Workflow:

```bash
$EDITOR settings.yml          # edit on host — IDE, syntax highlighting, git diff
docker compose restart        # SearxNG re-reads settings on startup
```

To see the **effective merged config** (override + inherited defaults):

```bash
docker exec searxng /usr/local/searxng/.venv/bin/python -c \
  "from searx import settings; \
   print('formats   :', settings['search']['formats']); \
   print('engines   :', len(settings['engines'])); \
   print('instance  :', settings['general']['instance_name'])"
```

The bind mount sits **on top of** the `searxng-config` volume mount, so
`settings.yml` lives in git while sibling files (favicon DB, future state)
stay in the volume out of git.

### Tradeoff vs vendoring the full file

- ✅ Tiny diffs — only your customizations are visible
- ✅ Future SearxNG releases auto-pick up new defaults (no merge pain)
- ⚠️ If upstream renames or removes a key you override, you find out at
  startup (container won't go healthy)
- ⚠️ Some defaults you might want to *see* (engine list, ratelimits) aren't
  in the file — use the `python -c` introspection above

## Volumes

Both volumes are declared `external: true` and live outside this compose
project's lifecycle:

| Volume          | Mounted at            | Holds                              |
| --------------- | --------------------- | ---------------------------------- |
| `searxng-config`| `/etc/searxng`        | `settings.yml` (with JSON format enabled), backups |
| `searxng-cache` | `/var/cache/searxng`  | favicon cache, rate-limiter state  |

`docker compose down` will NOT delete them. To wipe data explicitly:

```bash
docker volume rm searxng-config searxng-cache
```

## Why these specific overrides?

The slim `settings.yml` overrides exactly three keys, each for a concrete
reason:

| Key | Why we override |
|---|---|
| `server.secret_key` | Upstream default is the placeholder `ultrasecretkey`; SearxNG refuses to start with it |
| `search.formats` | Default is `[html]` only; jy-agent's `web_search` calls `?format=json` and gets **403** without `json` here |
| `general.instance_name` | Cosmetic — shows in browser tab when poking the UI |

## Reproduce from scratch

If volumes are missing, compose will fail with `external volume not found`.
Bootstrap them (the `settings.yml` in this repo is already correct, no
patching needed):

```bash
docker volume create searxng-config
docker volume create searxng-cache
docker compose up -d
```
