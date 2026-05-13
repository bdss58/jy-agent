---
name: wechat-mac-send
description: >-
  Send a message or image to a contact in WeChat for Mac via UI automation
  (clipboard + Quartz coordinate clicks + pixel row classification). Use this
  skill whenever the user asks to send something to a WeChat contact on macOS,
  forward an image to themselves via 文件传输助手 (File Transfer Helper),
  message a friend on WeChat from the agent, or otherwise drive WeChat for
  Mac from outside the app. TRIGGER on: "send X to <name> on WeChat",
  "send this image to WeChat", "WeChat 发送", "发给 文件传输助手",
  "forward to my WeChat", "post to WeChat Mac". DO NOT TRIGGER on: WeChat
  web/mobile, WeChat work/企业微信, reading existing WeChat messages
  (the AX tree exposes nothing — there is no read path), or any other
  Mac-app automation (write a new skill or use the macos tool modules
  directly).
metadata:
  author: jy-agent
  version: "1.0"
  platform: macos
  app: WeChat for Mac
---

# WeChat for Mac — Send to a Contact

WeChat for Mac is a custom render canvas: no AppleScript dictionary, AX tree
exposes only the 3 traffic-light buttons. The only reliable automation path
is **clipboard → ⌘F search → pixel-structural row detection → Quartz
coordinate click → clipboard reload → ⌘V → user-confirmed Return**.

All the heavy lifting lives in `jyagent.tools.macos.*`. This skill is the
checklist that orchestrates them and surfaces every gotcha at decision time.

> **Safety rail (non-negotiable)**: NEVER press the final Return on the
> agent's own initiative. After staging the message/image, take a
> screenshot, verify the chat header text matches the intended recipient,
> show the user a summary, and wait for explicit "send" / "go" / "yes"
> before pressing key code 36.

## Decision Tree

```
What does the user want to send?
│
├─ Text message to a contact          → Step 1, then 2-5, skip image-paste branch
├─ Image to a contact (incl. self)    → Full pipeline below
├─ Image to 文件传输助手 (themselves)  → Full pipeline (this is the canonical happy path)
└─ Forward an existing message        → Out of scope — no read path exists yet.
                                        Tell the user and stop.
```

## Step 0 — Verify Permissions (once per Mac)

macOS TCC has **two separate buckets** — both required:

1. System Settings → Privacy & Security → **Automation** → terminal app
   (Ghostty) → toggle ON for "WeChat" and "System Events".
2. System Settings → Privacy & Security → **Accessibility** → terminal app
   → toggle ON.

Quick check:

```bash
osascript -e 'tell application "System Events" to get name of first application process whose frontmost is true'
```

If this returns `-25211: 不允许辅助访问`, Accessibility is missing. If it
hangs or returns a permission error mentioning "automation", Automation is
missing. Tell the user which one and stop until they enable it.

## Step 1 — Prepare the Payload

### Image case

Ensure the image exists at a stable path. If the user just generated one
(e.g. via gpt-image), save it locally first:

```bash
ls -la /path/to/image.png   # must exist; size > 0
```

### Text case

The text itself is the payload — nothing to prepare.

## Step 2 — Bring WeChat to the Front

```bash
.venv/bin/python -m jyagent.tools.macos.keys activate WeChat
sleep 0.5
.venv/bin/python -m jyagent.tools.macos.keys keystroke 1 --cmd   # ⌘1 = Chats tab
sleep 0.3
```

## Step 3 — Open the Search Panel and Type the Contact Name

```bash
.venv/bin/python -m jyagent.tools.macos.clipboard set-text "<contact name>"
.venv/bin/python -m jyagent.tools.macos.keys keystroke f --cmd   # ⌘F
sleep 0.4
.venv/bin/python -m jyagent.tools.macos.keys keystroke v --cmd   # ⌘V (paste name)
sleep 1.6                                                         # let dropdown render
```

> **Trap — DO NOT press Return here.** The top result in the dropdown is
> a *query suggestion* that opens 问一问 (Q&A search), NOT the contact.
> The real contact lives further down, under a grey "功能" section header.
> Coordinate-click is the only reliable way to land on the right row.

## Step 4 — Locate the Real Contact Row

The ⌘F search opens a separate small window (~368×518 logical points,
typically anchored around screen position (222, 92) but verify per session).

Screenshot the panel and run the classifier:

```bash
# 1. Find the search-panel window bounds. WeChat exposes the front window's
#    position/size via System Events even though it hides everything else.
osascript -e '
  tell application "System Events" to tell process "WeChat"
    set p to position of front window
    set s to size of front window
    return (item 1 of p as text) & "," & (item 2 of p as text) & "," ¬
         & (item 1 of s as text) & "," & (item 2 of s as text)
  end tell'
# → "222,92,368,518"  (or similar; capture these as $PX $PY $PW $PH)

# 2. Capture and classify.
.venv/bin/python -m jyagent.tools.macos.screencap region $PX $PY $PW $PH /tmp/wx-search.png
.venv/bin/python -m jyagent.tools.macos.canvas_rows /tmp/wx-search.png --only contact --json
```

The JSON output lists every band the classifier labels `contact`. Pick the
**first** one (top-most real contact under the "功能" header).

Compute screen coordinates from the band:

```python
# image coords come from the JSON above (canvas_rows runs at native Retina)
from jyagent.tools.macos.canvas_rows import image_y_to_screen_y

PANEL_PX, PANEL_PY = 222, 92          # from the osascript above (LOGICAL coords)
PANEL_PW = 368                         # logical width of panel
band = first_contact_band              # the JSON entry from canvas_rows

# X: click roughly mid-panel (the contact row spans full width).
click_x = PANEL_PX + PANEL_PW / 2
# Y: image_y is in @2x pixels; image_y_to_screen_y handles the /2 scaling.
click_y = image_y_to_screen_y(band["y_center"], PANEL_PY)
```

In practice the proven values from the 2026-05-13 session were
`click_x = 406`, `click_y = 365` — but DO NOT hardcode them, re-derive
each time. The panel position varies with display, theme, and user resize.

## Step 5 — Click the Contact and Verify the Header

```bash
.venv/bin/python -m jyagent.tools.macos.mouse click "$CLICK_X" "$CLICK_Y"
sleep 0.8
# Capture the main chat window and read the header strip.
.venv/bin/python -m jyagent.tools.macos.screencap full /tmp/wx-after-click.png
```

**Verify** the chat header shows the intended recipient. Two options:

1. Pixel/OCR — extract the top header strip (~y 0..80 logical) and read.
2. If unsure, **ask the user** to confirm the chat header before proceeding.

If the header does NOT match: press Escape (`keycode 53`) to back out and
restart from Step 3 with a more specific search query. Never proceed on
ambiguity.

## Step 6 — Stage the Payload

### Image case

```bash
# IMPORTANT: the search text we pasted in Step 3 clobbered any prior
# clipboard contents. Always reload the image RIGHT BEFORE ⌘V.
.venv/bin/python -m jyagent.tools.macos.clipboard set-image /path/to/image.png
.venv/bin/python -m jyagent.tools.macos.clipboard verify-image   # exit 0 iff PNGf present
.venv/bin/python -m jyagent.tools.macos.keys keystroke v --cmd    # ⌘V → image stages in input
sleep 0.6
```

### Text case

```bash
.venv/bin/python -m jyagent.tools.macos.clipboard set-text "<message body>"
.venv/bin/python -m jyagent.tools.macos.keys keystroke v --cmd
sleep 0.3
```

## Step 7 — Final Verification and User Confirmation

```bash
.venv/bin/python -m jyagent.tools.macos.screencap full /tmp/wx-staged.png
```

Show the user a summary that includes:

- The chat header (recipient name) verified in Step 5.
- A confirmation that the payload is staged (image thumbnail visible in
  input field, or text visible).
- A request: "Reply 'send' to commit, anything else to abort."

**WAIT** for the user's reply. Do not loop, do not "interpret" silence as
confirmation. This is the safety rail that has saved every wrong-recipient
incident.

## Step 8 — Send

Only after explicit confirmation:

```bash
.venv/bin/python -m jyagent.tools.macos.keys keycode 36   # Return
sleep 0.5
.venv/bin/python -m jyagent.tools.macos.screencap full /tmp/wx-sent.png
```

Report back to the user with the post-send screenshot.

## Gotchas Quick Reference

| Symptom | Cause | Fix |
|---|---|---|
| `-25211: 不允许辅助访问` on any osascript | Accessibility TCC missing | System Settings → Privacy → Accessibility → toggle terminal ON |
| osascript hangs on first call | Automation TCC missing | System Settings → Privacy → Automation → toggle WeChat + System Events |
| `⌘F` then Return opens 问一问 Q&A page | Top result is a query suggestion, not a contact | Use coordinate click on the row labeled `contact` by canvas_rows, NEVER Return |
| ⌘V pastes plain text instead of image | Clipboard was overwritten by the search-text paste | Reload with `clipboard set-image` IMMEDIATELY before ⌘V |
| Down-arrow nav lands on unpredictable rows | Suggestion rows have different Enter behavior than contact rows | Bypass arrow nav; always coordinate-click |
| `front window` size is 368×518 not 1190×892 | The search panel is now the front window | Switch back to main-window bounds after dismissing search |
| Wrong contact (e.g. 公众号 instead of 文件传输助手) sent the image | Skipped header verification + user confirmation | Always do Steps 5 and 7. No exceptions. |
| Vision API timeouts mid-session | Codex `-i` flag hits 5-min upstream timeouts | Don't put vision on the critical path; canvas_rows is deterministic |

## What This Skill Does NOT Do

- **Read messages** — AX tree is empty. No read path exists. If the user
  needs to react to an incoming message, they must initiate.
- **Manage group memberships, contacts, settings** — out of scope.
- **Work on WeChat Web / Mobile / 企业微信** — different app, different
  surface. Write a separate skill if needed.
- **Handle multi-monitor / non-Retina displays** — calibrated for @2x
  Retina. Update `DEFAULT_RETINA_SCALE` in `screencap.py` and the
  `image_*_to_screen_*` `scale=` param if you ever run on a 1x display.

## See Also

- `jyagent/tools/macos/canvas_rows.py` — the pixel classifier (pure PIL)
- `jyagent/tools/macos/mouse.py` — Quartz click synthesis
- `jyagent/tools/macos/clipboard.py` — PNGf clipboard helpers
- `jyagent/tools/macos/keys.py` — AppleScript keystroke + key-code dispatch
- `jyagent/tools/macos/screencap.py` — region/full screenshot wrappers
- `tests/test_macos_canvas_rows.py` — classifier behavior tests
- `data/memory/topics/wechat-mac-automation.md` — historical playbook
  (this skill is the operational form of it)
