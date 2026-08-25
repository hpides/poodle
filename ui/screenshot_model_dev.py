import asyncio
from playwright.async_api import async_playwright

WIDTH = 1330
HEIGHT = 1000

async def screenshot():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=3
        )
        # Force light mode: the page reads this from localStorage on load
        await context.add_init_script(
            "localStorage.setItem('poodle-theme', 'light');"
        )
        page = await context.new_page()
        await page.goto("http://localhost:8080", wait_until="networkidle")

        # Navigate to model-dev view first
        await page.click("button:has-text('Details')")
        await page.wait_for_load_state("networkidle")

        # Click Start All and wait for runs to populate
        await page.click("button:has-text('Start All')")
        await asyncio.sleep(10)

        await page.screenshot(
            path="screenshot_model_dev.png",
            full_page=False
        )

        await page.pdf(
            path="screenshot_model_dev.pdf",
            width=f"{WIDTH}px",
            height=f"{HEIGHT}px",
            print_background=True
        )

        await browser.close()
        print("Done!")

asyncio.run(screenshot())
