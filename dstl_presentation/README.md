# Dstl sponsor presentation: *Intelligence from imperfect sensors*

A 30-minute reveal.js deck (27 slides) for the Dstl sponsor review, October 2026. Most of the time goes on the
Experiment 3 (RIoT v5) results. Every chart is a standalone D3 page in `figures/`, drawn from real data in
this repository. The palette is the GOV.UK Design System colour set, and Arial stands in for GDS Transport.

## Run it

```bash
cd dstl_presentation
python3 -m http.server 8000          # or: npm run serve
# open http://localhost:8000
```

You can also open `index.html` straight from disk, because the vendored runtime and the `figures/data/*.js` caches
need no server.

| Key | Action |
|---|---|
| `→` / `Space` | next fragment or slide. Fragments also step the embedded figures. |
| `S` | speaker view, with notes and per-slide timings in `[m:ss]` |
| `Esc` | slide overview |
| `F` | full screen |

**Live dashboard (slide 12).** This slide is a full-screen interactive iframe of `http://localhost:3000`. Start the
dashboard first (`docker compose up frontend`). If it isn't running the slide is blank; slide 18 shows the same
Pilgrim Street outage as a static figure.

## Regenerate the figure data

```bash
python3 dstl_presentation/scripts/build_figure_data.py
```

The script needs `pandas`, `numpy` and `pyarrow`. It reads:

| Source | Used for |
|---|---|
| `docs/experiments/riot_v5/data/runs.csv` | all Experiment 3 charts (103 runs) |
| `artifacts/payload_staging/runs/` | real-outage example, outage awareness, network map |
| `artifacts/quality_full_v2/` | Experiment 1 |
| `artifacts/spatial_full/aggregated/` | Experiment 2 |
| `data/traffic_20230108_20260608.parquet` | availability heat map, failure-mode snippets, weekly profiles, gap statistics |
| `logs/scaling/suite.log` | density-scaling points |

**The density-scaling suite is still running.** N = 35 and N = 45 show as "running" on slide 23. Re-run the script
when they finish and the chart picks them up automatically.

## Layout

```
index.html                 slides and speaker notes
css/deck.css               theme (GOV.UK colours, layout primitives)
js/deck.js                 reveal.js set-up, footer progress, fragment → figure bridge
figures/*.html             one interactive figure per file (open any of them on its own)
figures/lib/fig.{css,js}   shared figure styling, tooltip, step control, model colours
figures/data/*.js          generated data caches (window.DATA_*)
scripts/build_figure_data.py
scripts/vendor.mjs         copies reveal.js and d3 from node_modules into vendor/
vendor/                    vendored reveal.js 5.2 and d3 7.9 (offline)
```

Figures build up in step with slide fragments: `js/deck.js` posts the fragment count to each iframe. On its own a
figure shows its final state. Add `?step=N` to see a particular stage, or `?zoom=1` to turn off the projection
zoom.

Each model keeps the same colour on every slide: V5F blue, V4R orange, E2.1 turquoise, V5F without unified
encoding purple, P2B light blue, E3.3d brown, and the naïve baseline dark grey.

## Before presenting: fill in or check

- **Slide 2:** names, roles and interests for the supervisors and the Dstl contact (yellow placeholders).
- **Slide 26:** publication statuses, the MDPI paper's full title and the timeline dates (yellow placeholders).
- **Slide 27:** contact email.
- **Slide 4:** confirm the policy references (Defence AI Strategy 2022, JSP 936) suit the audience.
- **Slide 5:** the benchmark missingness rates are as reported by GRIN (Cini et al., 2022) and SAITS (Du et al.,
  2023). Check them against the papers if they are likely to be challenged.
- **Slide 9 notes:** the Heber Street surge (Sun 23 Feb 2025, about 16:00) looks like a crowd leaving St James'
  Park. Confirm the fixture before you say so.
- All Experiment 3 numbers are **single-seed** (noise floor about ±3%), as the deck says on slide 15.

## Timing (target 30 min)

| Slides | Section | Time |
|---|---|---|
| 1–7 | Context: idea, defence relevance, state of the art, thesis | 8:00 |
| 8–10 | Data quality and Experiment 1 | 4:30 |
| 11–12 | Digital twin and live demo | 3:00 |
| 13–22 | Why a richer model, RIoT, Experiment 3 findings | 12:30 |
| 23–27 | Scalability, summary, implications, outputs | 4:15 |

The per-slide timings in the notes add up to about 32 minutes. To fit 30, drop slide 19 (the
leave-one-out results) or shorten the demo.
