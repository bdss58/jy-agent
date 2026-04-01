# Browser Automation Workflows

## Login to a Website

```
1. navigate_page → login URL
2. take_snapshot → find form field UIDs
3. fill(uid, "username") → username field
4. fill(uid, "password") → password field
5. click(uid) → submit/login button
6. wait_for("Dashboard") → confirm login succeeded
7. take_snapshot → verify logged-in state
```

**Gotchas:**
- Some sites have CAPTCHA — you may need user intervention
- Some sites have 2FA — wait for the user to complete it
- Cookie consent popups may block the form — dismiss them first

## Scrape a Data Table

```
1. navigate_page → page with the table
2. wait_for("expected header text") → confirm table loaded
3. take_snapshot → see table structure
4. evaluate_script → extract data via JS:
   
   // Example: extract all rows from a table
   const rows = document.querySelectorAll('table tbody tr');
   return Array.from(rows).map(row => {
     const cells = row.querySelectorAll('td');
     return Array.from(cells).map(c => c.textContent.trim());
   });
```

**Gotchas:**
- Tables may be lazily loaded — scroll down first with `evaluate_script("window.scrollTo(0, document.body.scrollHeight)")`
- Virtual scroll tables only render visible rows — you'll need to scroll and collect incrementally
- Some tables are actually CSS grid/flexbox, not `<table>` elements — inspect the snapshot

## Fill a Multi-Page Form

```
For each page:
  1. take_snapshot → identify form fields on this page
  2. fill(uid, value) → fill each field
  3. take_screenshot → visual verification (optional)
  4. click(uid) → "Next" button
  5. wait_for("Step 2") → confirm page transition
  
Final page:
  6. click(uid) → "Submit" button
  7. wait_for("Success") → confirm submission
```

**Gotchas:**
- Form validation errors may appear — check the snapshot after clicking Next
- Some forms auto-save, others don't — be careful with page refresh
- Date pickers and custom dropdowns need special handling (see below)

## Handle Custom UI Components

### Date Pickers
```
1. click(uid) → open the date picker
2. take_snapshot → see the calendar
3. click(uid) → navigate to the right month/year
4. click(uid) → click the target date
```
Or shortcut: `fill(uid, "2024-01-15")` sometimes works on the underlying input.

### Autocomplete/Search Dropdowns
```
1. type_text(uid, "search term") → trigger autocomplete (use type_text, not fill!)
2. wait_for("expected option text") → wait for suggestions
3. take_snapshot → find the option UID
4. click(uid) → select the option
```

### File Upload
```
1. take_snapshot → find the file input UID
2. fill(uid, "/path/to/file") → set the file path (works on <input type="file">)
```

## Handle Pagination

```
results = []
while True:
  1. take_snapshot → extract data from current page
  2. evaluate_script → collect data, append to results
  3. take_snapshot → look for "Next" button
  4. If no "Next" button or it's disabled → break
  5. click(uid) → click "Next"
  6. wait_for("Page N") → confirm page loaded
```

## Download Files

```
1. navigate_page → page with download link
2. take_snapshot → find the download link UID
3. click(uid) → trigger download
4. evaluate_script → check download via:
   
   // Get the href to download manually
   document.querySelector('a.download-link').href
```

Note: Browser downloads go to Chrome's default download directory. For controlled downloads, extract the URL and use `run_shell("curl -o output.file 'URL'")` instead.

## Chinese Website Tips (淘宝, 知乎, 百度, etc.)

- **Login walls are common** — many Chinese sites require login to view content
- **Anti-bot is aggressive** — avoid rapid page transitions, add `evaluate_script("await new Promise(r => setTimeout(r, 2000))")` between actions
- **Character encoding** — snapshot handles this correctly, but evaluate_script results may need UTF-8 handling
- **Mobile versions** — sometimes easier: resize_page(375, 812) and navigate to m.xxx.com
