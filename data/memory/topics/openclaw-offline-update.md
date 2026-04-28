---
created: 2026-04-28T12:32:02+08:00
updated: 2026-04-28T15:01:21+08:00
---
# OpenClaw Offline Update Process

Offline bundle shipping OpenClaw + plugins to customer hosts (no internet).

## Bundle layout (post-2026.4.28 generalization)

Bundle root ships ONE script (`openclaw.sh`) with subcommands. The image
itself is **self-validating**: it carries a preflight runner that both the
build-host self-check and the customer-host install/update preflight
invoke. Single source of truth — no duplicate assertion lists between
build script and customer script.

```
manifest.sh                  # build-host: declares BUNDLE_NAME, TAG, COMPONENTS[]
build-and-export.sh          # sources manifest.sh, builds + bundles
preflight.sh                 # /app/preflight.sh — runner inside the image
preflight.d/<NN>-<name>.sh   # /app/preflight.d/* — one check per component
openclaw.sh                  # customer-host: load/add/update/list/rollback
Dockerfile                   # COPYs preflight.sh + preflight.d/ into runtime stage
```

## Customer-host commands

```
./openclaw.sh load                 # docker load + arch check + --network none preflight
./openclaw.sh add <name> <port>    # create + start instance under ./instances/<name>/
./openclaw.sh update [<name>|--all]  # flip OPENCLAW_IMAGE per-instance, atomic recreate
./openclaw.sh list                 # instances + image tags + container status
./openclaw.sh rollback <name>      # restore latest .env.bak.<ts> + recreate
```

Always multi-instance — each instance is its own compose project under
`./instances/<name>/` with its own `.env`, port, container name
(`openclaw-wecom-<name>`), and `data/openclaw/` bind mount. The compose
file is symlinked from the bundle root, so a new bundle brings a new schema.

## Adding a new component to the image (the killer feature)

This is now a **single-place** change. Previously edits in three files
that silently drifted; now:

1. Edit `Dockerfile` — install commands (apt / pip / npm / curl …).
2. (Optional) pin a version in `manifest.sh`:
   `COMPONENTS+=("MYTHING:1.2.3")` → auto-passed as
   `--build-arg MYTHING_VERSION=1.2.3` and recorded in bundle MANIFEST.
   Dockerfile must `ARG MYTHING_VERSION` to consume.
3. Drop `preflight.d/<NN>-<name>.sh` — small bash script,
   `set -euo pipefail`, exit non-zero on failure. Auto-discovered by
   `/app/preflight.sh` lexicographically.

NO changes to `build-and-export.sh` or `openclaw.sh` required. The new
check shows up in build self-check output AND every customer-host install
preflight automatically. Same code, both sides.

## Customer upgrade flow
1. Build host: `TAG=<NEW> ./build-and-export.sh` (must bump TAG, otherwise
   `update` is a no-op).
2. `scp dist/<bundle-name>-offline-<NEW>.tar.gz customer:/root/`
3. Customer: `tar xzf …`, `cd <new-bundle>`, `mv ../<old-bundle>/instances ./`,
   `sudo ./openclaw.sh update --all`.
4. `update` loads new image + runs preflight ONCE upfront. If preflight
   fails, no instance is touched. Per-instance: backs up `.env` →
   `.env.bak.<ts>`, flips `OPENCLAW_IMAGE`, `compose up -d`, waits for
   /healthz. On failure, old image stays loaded for `rollback`.

## Bundle metadata files (in the tarball root)
- `IMAGE_REF` — full image:tag ref
- `BUNDLE_NAME` — variant identity (e.g. "openclaw-wecom")
- `VERSION` — same as TAG
- `MANIFEST` — one `NAME=VERSION` per line for every pinned component
- `PLATFORM` — `linux/arm64` or `linux/amd64`

## Variants
Different image flavors live in sibling manifest files:
```
MANIFEST=manifest-foo.sh ./build-and-export.sh
```
Same Dockerfile, different `COMPONENTS[]` / `BUNDLE_NAME` / build-args.

## Known gotchas

### Build artifact staleness
When extending the bundle AFTER a build, never just patch the build script
and assume the artifact is updated. Re-run `./build-and-export.sh`, then
verify:
```
tar tzf dist/<bundle>-offline-<tag>.tar.gz | grep <new-file>
```
Common failure: `scp`'ing the new file directly into an extracted dir on
the remote host hides the bug — that copy works, but the local tarball
is still stale.

### docker-compose.yml schema changes don't auto-propagate
The instance's compose file is a symlink to the bundle root's
`docker-compose.yml`. If you `mv ../old/instances ./` into a new bundle,
the symlinks are still valid (relative path `../../docker-compose.yml`
resolves to the NEW bundle's compose). But if customer kept the OLD
bundle dir and just changed OPENCLAW_IMAGE by hand, they'd miss new
services / volumes. `openclaw.sh update` doesn't yet detect schema drift.

### Legacy single-install layout (pre-2026.4.28 bundles)
Old bundles used a single install at the bundle root (no `instances/`
layer). To migrate to the new layout:
```
mkdir -p instances/prod
mv data docker-compose.yml .env instances/prod/
# Append to instances/prod/.env:
#   COMPOSE_PROJECT_NAME=openclaw-wecom-prod
#   INSTANCE_CONTAINER=openclaw-wecom-prod
#   INSTANCE_DATA_DIR=./data/openclaw
ln -sf ../../docker-compose.yml instances/prod/docker-compose.yml
```

## Related environment
- Test cluster: `wan2` (`wan2.think-force.com`), Kubernetes.
- `kubectl port-forward` binds 127.0.0.1 only by default — use
  `--address 0.0.0.0` to expose externally.
