# Pipeline schematics

Two high-level, non-technical schematics of what the RIoT pipeline sets out to achieve. Both are for use in
PDF reports.

| File | Audience | Framing |
|---|---|---|
| `output/defence.pdf` | Defence (Dstl) | Assured situational awareness under DDIL conditions; outputs mapped to Endsley's levels |
| `output/urban.pdf` | Urban science | A city digital twin that keeps working through outages; likely users |

Each schematic is also exported as `.svg` (editable vector) and `.png` (2× raster preview, 2400 × 2040).

Both tell the same three-stage story:
1. **Imperfect sensor data.** A sensor map shows intermittent, silent, lost and missing sensors.
2. **A resilient model.** It reads the gaps, borrows from neighbours and knows when it is unsure.
3. **A complete picture.** Every location has an estimate, with a confidence halo and sensor-health flags.

Underneath is a "graceful degradation" panel and an audience-specific box. The sensor layout uses real station
positions from the Newcastle network.

## Using in a report

- **LaTeX:** `\includegraphics[width=\linewidth]{schematics/output/defence.pdf}`
- **Word:** insert the `.svg` (Office 365) or the `.png`.
- The artwork is 1200 × 1020 (aspect ≈ 1.18 : 1). Text is Arial at 12–30 px. At full A4 landscape page width
  that comes to about 9–22 pt. At portrait text width (≈ 16 cm) the smallest notes are about 5 pt, so a
  landscape page or a full-width float is recommended.

## Editing and rebuilding

All wording is in the `SCHEMATIC_CONTENT` block of `defence.html` and `urban.html`. Layout and drawing are
shared in `src/schematic.js`, and the sensor positions and failure states are in `src/nodes.js`.

```bash
python3 schematics/build.py      # needs Python Playwright + Chromium
```

You can also open either HTML file in a browser to preview it.
