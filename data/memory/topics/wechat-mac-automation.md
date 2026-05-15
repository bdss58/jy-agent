---
created: 2026-05-13T18:53:27+08:00
updated: 2026-05-13T18:53:27+08:00
---
# WeChat for Mac Automation Playbook

Tested 2026-05-13. WeChat for Mac is hostile to automation: custom render canvas,
no AppleScript dictionary, AX tree exposes only window-chrome buttons. The path
that actually works is **clipboard + Quartz coordinate clicks + pixel structural
analysis**, with vision as a nice-to-have not a load-bearing primitive.

## Permissions (one-time, on user's Mac)

Two SEPARATE TCC buckets — both required:

1. **System Settings → Privacy & Security → Automation** → terminal app (Ghostty)
   → toggle "WeChat" and "System Events" ON.
2. **System Settings → Privacy & Security → Accessibility** → terminal app (Ghostty)
   toggle ON. Without this, `tell process "WeChat"` AX queries fail with
   error `-25211: "不允许辅助访问"`.

Verify with: `osascript -e 'tell application "System Events" to get name of first
application process whose frontmost is true'` — should return in <1 s.

## What AppleScript Can Do (very little)

- `tell application "WeChat" to activate` — works (foreground the app)
- `set frontmost to true` — works
- Send keystrokes (`keystroke "f" using {command down}`, `key code 36` for Return,
  `key code 125` for Down, `key code 53` for Escape) — works
- Read window position/size/count — works
- Read window title — usually empty (Tencent does not set per-chat titles)
- **Cannot** read chat header text, sidebar items, message content, input field
  contents — AX tree has only 3 traffic-light buttons.

## Finding a Contact: The 文件传输助手 Trap

`⌘F` opens a **separate small search window** (368×518 at pos ~222,92), NOT a
sidebar overlay. The results dropdown is structured as:

```
[search input field]
Row 1: 文件传输助手               ← query suggestion (opens 问一问 search page on Enter!)
Row 2-5: 文件传输助手{suffix}     ← more query suggestions
[section header: "功能"]          ← grey small text, ~12 logical px
Row 6: 文件传输助手               ← THE REAL CONTACT (tall 2-line row with avatar)
[section header: "群聊"]
Row 7+: group chats
```

**Pressing Enter on the highlighted top result opens 问一问 (Q&A search results),
NOT the contact.** Down-arrow does highlight rows but Enter still triggers the
suggestion behavior on suggestion rows. The reliable path is **coordinate click**
on the real contact row.

## Pixel-Structural Row Detection (PIL only, no vision)

When vision is flaky, scan the search-panel screenshot to find rows
deterministically:

```python
from PIL import Image
img = Image.open(screenshot_path).convert('RGB')
W, H = img.size  # Retina @2x of logical panel

TEXT_X0, TEXT_X1 = 110, 700  # skip icon column, focus on text column

def row_min_text(y):
    return min(min(img.getpixel((x, y))) for x in range(TEXT_X0, TEXT_X1, 3))

# Find text bands (rows with any dark text pixel < 180)
text_rows = [y for y in range(H) if row_min_text(y) < 180]
# Group consecutive rows (gap <= 3) into bands
# For each band, classify by:
#   - has_green: pixel matching (g>150, r<130, b<130) → keyword highlight
#   - has_black: r<80, r==g==b → label/contact-name text
#   - has_grey:  120<r<200, r==g==b → section header text
#   - band_height: 22 image-px (~11 logical) = single-line suggestion row
#                  67-72 image-px (~33-36 logical) = 2-line contact/chat row
```

Signatures observed for WeChat search panel:
- **Search input**: h≈21, grey-only (the pasted text on grey field bg)
- **Query suggestion row**: h≈22, has green + has black (query in green, suffix in black)
- **Section header (功能/群聊/相关搜索)**: h≈12-20, grey-only, no green
- **Real contact row**: h≈67-72 (two-line), has green name + grey subtitle

Convert image_y → screen_y: `screen_y = panel_origin_y + image_y / 2`
(divide by 2 because Retina, panel_origin_y was 92 in our session).

## Mouse Click via Quartz

```python
import Quartz, time
def click(x, y):
    move = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved,
                                          (x, y), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, move); time.sleep(0.15)
    down = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseDown,
                                          (x, y), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down); time.sleep(0.05)
    up = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventLeftMouseUp,
                                        (x, y), Quartz.kCGMouseButtonLeft)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
```

Install: `.venv/bin/python -m pip install pyobjc-framework-Quartz`
(after bootstrapping pip: `.venv/bin/python -m ensurepip --upgrade`).

## Image to Clipboard (for ⌘V Paste as Image)

```bash
osascript -e 'set the clipboard to (read (POSIX file "/path/img.png" as alias) as «class PNGf»)'
```

Then ⌘V in WeChat input pastes as a real image attachment (not a file path).
Verify clipboard with `osascript -e 'clipboard info'` — should list
`«class PNGf», JPEG picture, TIFF picture, ...`.

**WARNING**: Setting clipboard to text (e.g. for pasting search query) overwrites
the image. Always re-load image to clipboard AFTER any text-paste step.

## End-to-End Recipe: Send Image to a Contact via Search

```
0. Verify permissions (Automation + Accessibility) once
1. set the clipboard to "<contact name>"  (text)
2. activate WeChat; ⌘1 (Chats tab); ⌘F (search); ⌘V (paste query)
3. delay 1.6s for dropdown render
4. screencapture -x -R<search-panel-bounds> /tmp/search.png
5. PIL scan → find the tall (h≈70 img-px) row with green-name signature
   → compute screen coords (center of panel x, band_center_y/2 + panel_y)
6. Quartz click at those coords
7. delay 1s; screencapture main window; crop header strip;
   verify with vision (or pixel OCR) that header == "<contact name>"
8. Re-load image to clipboard as PNGf
9. ⌘V to paste image into input
10. screencapture + verify image is staged (preview thumbnail visible)
11. STOP — show user a summary and wait for explicit "send" confirmation
12. key code 36 (Return) to send
13. screencapture to verify message appears in chat history
```

## Safety Rail (Non-Negotiable)

For irreversible actions (send/delete/post), the agent ALWAYS:
1. Screenshots immediately before the action
2. Verifies recipient/target identity (chat header, button label)
3. Pauses and shows user the verification result
4. Only proceeds on explicit user "send" / "go" / "yes"

Skipping this caused the first failure of this session (blindly pressing Enter
opened 公众号 instead of 文件传输助手). Trust verification > trust automation.

## Pitfalls / Known Failures

- **Vision endpoint flakiness**: Codex's `codex exec -i img.png` is great when
  it works but routinely hits 5-min upstream timeouts. Don't put it on the
  critical path; treat it as a fallback for ambiguous cases.
- **WeChat window-count mismatch after ⌘F**: front window becomes the search
  panel (small 368×518), not the main 1190×892 chat window. Use
  `front window` for screenshots, but switch back to main-window bounds after
  the search panel is dismissed.
- **⌘F search Down arrow appears unreliable** — vision claimed no highlight
  after 1 Down, then 6 Down arrows landed... somewhere ambiguous. Bypass
  arrow navigation entirely; coordinate-click is more reliable.
- **Esc closes search panel** but may also close the chat — use sparingly.
