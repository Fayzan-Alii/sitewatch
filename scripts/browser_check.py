"""Real dashboard navigation and API-backed rendering check."""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=True, channel="chromium", args=["--no-sandbox"]
    )
    page = browser.new_page(
        viewport={"width": 1440, "height": 1050}, device_scale_factor=1
    )
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8000", wait_until="networkidle")
    page.get_by_text("System connected", exact=False).wait_for()
    pages = []
    for name in ["Monitor", "Events", "Human Review", "Performance"]:
        page.get_by_role("button", name=name, exact=False).first.click()
        page.wait_for_timeout(400)
        page.screenshot(
            path=f"results/dashboard-{name.lower().replace(' ', '-')}.png",
            full_page=True,
        )
        pages.append({"page": name, "title": page.locator("h1").inner_text()})
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Monitor", exact=True).click()
    page.screenshot(path="results/dashboard-mobile.png", full_page=True)
    overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
    output = {
        "pages": pages,
        "console_errors": errors,
        "mobile_horizontal_overflow": overflow,
    }
    Path("results/browser-verification.json").write_text(json.dumps(output, indent=2))
    print(output)
    browser.close()
    assert not errors and not overflow
