import asyncio
from playwright.async_api import async_playwright

WIDTH = 1440
HEIGHT = 2000

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

        # PNG: full_page=True ignores HEIGHT (captures all content)
        # Use full_page=False to respect HEIGHT as a clip
        await page.screenshot(
            path="screenshot.png",
            full_page=False  # clips to viewport HEIGHT
        )

        # PDF: viewport is irrelevant — set height explicitly here
        await page.pdf(
            path="screenshot.pdf",
            width=f"{WIDTH}px",
            height=f"{HEIGHT}px",   # <-- this is what controls PDF height
            print_background=True
        )

        await browser.close()
        print("Done!")

asyncio.run(screenshot())