#!/usr/bin/env python3
"""Export the schematics to vector PDF, standalone SVG and a 2x PNG preview.

    python3 schematics/build.py      # writes schematics/output/{defence,urban}.{pdf,svg,png}

Needs Python Playwright with Chromium (`pip install playwright && playwright install chromium`).
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
NAMES = ["defence", "urban"]
W, H = 1200, 1020  # must match the viewBox in src/schematic.js


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for name in NAMES:
            page = await browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.goto((HERE / f"{name}.html").as_uri())
            await page.wait_for_timeout(300)
            if errors:
                raise RuntimeError(f"{name}: {errors}")
            svg = await page.eval_on_selector("svg", "s => s.outerHTML")
            (OUT / f"{name}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg)
            await page.screenshot(path=str(OUT / f"{name}.png"), clip={"x": 0, "y": 0, "width": W, "height": H})
            # One page sized exactly to the drawing; text stays vector and selectable.
            await page.pdf(path=str(OUT / f"{name}.pdf"), width=f"{W}px", height=f"{H}px",
                           print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                           page_ranges="1")
            await page.close()
            print(f"wrote output/{name}.{{svg,pdf,png}}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
