# OpenClaw Offline Update Process

Notes on shipping offline image updates to customer hosts for OpenClaw.
**One-line pointer in MEMORY.md.** Detail here.

## Context
Jianyong ships OpenClaw to customer-hosted environments. Updates are
delivered as offline bundles (tarballs with images + `update-offline.sh`
installer + compose config).

## Known Issues / Lessons

### `update-offline.sh` preflight is hardcoded to image contents
Current preflight is tied to specific images. For generic updates, move
the preflight script **into the image itself** and have the installer
invoke it from the image. This decouples preflight logic from the installer
version.

### `docker-compose.yml` schema changes don't auto-propagate
When the new release changes the compose schema (new services, modified
volumes), existing customer deployments won't pick them up automatically.

Two viable patterns:
- Ship an `override.yml` and merge on the customer host.
- Detect `docker-compose.yml.new` and prompt / auto-apply on update.

### Build artifact staleness
**When extending a build artifact (tarball / bundle) AFTER it was
produced: never just patch the build script and assume the artifact is
updated.** Re-run the build, then verify the new file is inside:

```bash
tar tzf bundle.tgz | grep <new-file>
```

Common failure mode: `scp`'ing a new file directly into an extracted dir
on a remote host hides the bug — that copy works, but the local artifact
is still stale.

## Related Environment
- Test cluster: `wan2` (`wan2.think-force.com`), Kubernetes, `kubectl
  port-forward` on port 31555.
- Caveat: `kubectl port-forward` binds 127.0.0.1 only by default; use
  `--address 0.0.0.0` to expose externally.
