"""Capture dashboard screenshots for the dissertation using Playwright."""
import time
from playwright.sync_api import sync_playwright

from config import OUTPUTS_DIR

URL = "http://localhost:8530"


def wait_loaded(page, must_contain, timeout=180):
    """Wait until Streamlit has actually rendered real content."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            body = page.inner_text("body")
            running = page.query_selector('[data-testid="stStatusWidget"]')
            if must_contain in body and running is None:
                time.sleep(3)
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 1200})
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)

    # ---------- Tab 1 ----------
    ok = wait_loaded(page, "Predicted admission risk")
    print("tab1 initial render:", ok)

    # switch to a high-risk record (index 2382, risk 99.4%, 4 recommendations)
    box = page.query_selector('input[type="number"]')
    if box:
        box.fill("2382")
        box.press("Enter")
        time.sleep(4)
        wait_loaded(page, "Recommended community-care actions")
        print("switched to high-risk record 2382")

    page.screenshot(path=str(OUTPUTS_DIR / "dash1_individual.png"),
                    full_page=True)
    print("saved dash1_individual.png")

    tabs = page.query_selector_all('button[data-baseweb="tab"]')
    print("tabs found:", len(tabs), [t.inner_text().strip() for t in tabs])

    # ---------- Tab 2: Cohort ----------
    if len(tabs) >= 2:
        tabs[1].click()
        ok = wait_loaded(page, "Cohort risk overview")
        print("tab2 rendered:", ok)
        page.screenshot(path=str(OUTPUTS_DIR / "dash2_cohort.png"),
                        full_page=True)
        print("saved dash2_cohort.png")

    # ---------- Tab 3: Performance ----------
    tabs = page.query_selector_all('button[data-baseweb="tab"]')
    if len(tabs) >= 3:
        tabs[2].click()
        ok = wait_loaded(page, "Best model")
        print("tab3 rendered:", ok)
        page.screenshot(path=str(OUTPUTS_DIR / "dash3_performance.png"),
                        full_page=True)
        print("saved dash3_performance.png")

    browser.close()
print("Done.")
