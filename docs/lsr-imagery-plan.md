# Plan: imagery for the LSR test cases with GeoIPS and courier

Status: plan, nothing built yet.

## Goal

For each report in the golden set, produce the imagery a forecaster uses to
judge a hail report, following `Hail Report Evaluation Guide.pptx`, and in
addition the MRMS grids LSR-AI's own tools return. Uses:

- input for a vision model, or an LSR-AI-style agent, run on the golden set
- material for human reviewers, in the same layout as the deck
- checking LSR-AI's tool outputs, which are recorded in the two logs in
  `data/lsr-ai/`

Scope: the 1,044 coded reports in "Seans Reports" (1,041 hail, 3 wind; 904
Good, 140 Bad; 155 radar cases on 92 dates, May 2021 to Feb 2026), plus the
two LSR-AI examples.

## What the forecaster looks at

From the deck (7 and 9 July 2025; the slide titles say 2026, but the image
stamps are the correct dates):

| Deck | Imagery | Source | Made by |
|---|---|---|---|
| Slides 1, 12 | Overview: lowest-tilt reflectivity around the radar (~450 × 250 km), every hail (green square) and wind (blue circle) report for the day | NEXRAD Level II, closest radar; reports from the sheet | GeoIPS, new Level II reader |
| Slides 2–4, 13–15 | Environment: MUCAPE, effective bulk shear, effective SRH, freezing level, Significant Hail Parameter, Large Hail Parameter | SPC mesoanalysis | Downloaded (see below) |
| Slide 5 | Criteria: 50 dBZ height vs. melting level (Cavanaugh and Schultz 2012), 60 dBZ above 30 kft, storm-top divergence > 100 kt, rotation, BWER, CC/ZDR, TBSS | Reference table, not imagery | Goes in the prompt |
| Slides 6, 9, 16 | 4-panel at report time: reflectivity, velocity, ZDR, CC, ~60 × 40 km on the report | NEXRAD Level II lowest tilt | GeoIPS |
| Slides 7, 10 | Storm-top divergence: reflectivity + velocity at the tilt through the storm top | NEXRAD Level II upper tilt | GeoIPS |
| Text of 6, 9, 16 | 50 and 60 dBZ echo heights "at time of report and a few minutes before" | MRMS `EchoTop_50`, `EchoTop_60`, `H50_Above_0C`, `H60_Above_-20C`; melting level from MRMS `Model_0degC_Height` | GeoIPS, new MRMS reader |
| LSR-AI tools | `MESH_00.50`, `Reflectivity_-20C_00.50`, `MergedReflectivityQCComposite_00.50`, `VIL_00.50` and `VIL_Density_00.50` (see validation 1), `PrecipFlag_00.00` | MRMS | GeoIPS, new MRMS reader |

The forecaster also uses storm history (earlier reports from the same storm)
and the narrative (for example, "estimated from a photo"). Both come from the
sheet, not from imagery, and go in the case text.

### Data sources (checked)

| Source | Location | Covers all cases? | Size |
|---|---|---|---|
| MRMS | `s3://noaa-mrms-pds/CONUS/<product>/<YYYYMMDD>/MRMS_<product>_<YYYYMMDD-HHMMSS>.grib2.gz`, every 2 min | Yes, from 2021-05-24 (all products listed above) | 50 KB (MESH) to 1.8 MB (composite) per file |
| NEXRAD Level II | `s3://unidata-nexrad-level2/<YYYY>/<MM>/<DD>/<RADAR>/<RADAR><YYYYMMDD>_<HHMMSS>_V06`, every 4–6 min | Yes, from 2021-05-24 | ~12 MB per volume |
| SPC mesoanalysis | `https://www.spc.noaa.gov/exper/ma_archive/images_s4/<YYYYMMDD>/<HH>_<param>.gif`, hourly; `mucp`, `eshr`, `effh`, `fzlv`, `sigh`, `lghl` | Yes, but **national sector only** | ~30–75 KB |
| RAP analysis | `s3://noaa-rap-pds/rap.<YYYYMMDD>/rap.t<HH>z.awp130pgrbf00.grib2` | Yes | ~17 MB |

All are public, with no login. The sheet already holds the closest radar
(`Closest_Radar`) and its distance.

**SPC caveat.** The deck shows a regional SPC sector, but the archive keeps
only the national one (s4). Phase 1 crops the national GIF to the case
region, which gives the same contours at lower zoom. Rebuilding regional maps
means computing the parameters from RAP (GeoIPS reader + MetPy for effective
layer, SHIP and LHP), which is a lot of work and is left for later, as an
option.

## Courier layout

Three stages, as you asked: a monitor on the sheet, a dispatcher that
downloads, and a second monitor on the downloaded and unzipped data.

```
LSRGenAI Golden Test Set Cases.xlsx  ("Seans Reports")
        │
  [data monitor] lsr_sheet ─── one Datum per report row (new or changed)
        │                      file=<xlsx>, source=lsr_sheet, timestamp=<report time UTC>,
        │                      metadata={event_id, lat, lon, event, magnitude, radar, ...}
  [job builder]  filter_and_group  filters: {source: lsr_sheet}, files_per_job: 1
        │
  [dispatcher]   downloader (local_dispatcher)
        │        payload: python_payload
        │          scripts/lsr_fetch.py --sheet {{ files[0].file }}
        │            --event-id {{ files[0].metadata.event_id }} --time {{ files[0].timestamp }}
        │            --lat {{ files[0].metadata.lat }} --lon {{ files[0].metadata.lon }}
        │            --radar {{ files[0].metadata.radar }}
        │          fetch MRMS / Level II / SPC for the case window, gunzip,
        │          write to ready/<event_id>/..., write the completion marker last
        ▼
data/lsr-imagery/ready/
        │
  [data monitor] ready_watch (file_system_poller_watchdog, path: ready/)
        │        metadata-tools: [lsr_ready]  (source + timestamp from file names)
  [job builder]  filter_and_group  filters: {source: lsr_case_ready}, files_per_job: 1
        │
  [dispatcher]   renderer (local_dispatcher, later slurm_dispatcher)
                 payload: python_payload
                   scripts/lsr_render.py {{ files[0].file }} --time {{ files[0].timestamp }}
                   GeoIPS: read → per-case sector → products → PNG / netCDF / table
```

Everything runs in one courier `Service` (`configs/courier/lsr-imagery.yaml`).
The two job builders filter on `source`, so each takes only its own
monitor's datums.

Courier makes each datum available to Jinja as `files[i]` (its `to_dict()`
form). That covers `file`, `source`, `timestamp` (an ISO 8601 UTC string),
`metadata`, `data` and the rest; see `Payload._template_context` in
`courier/interfaces/payloads.py`. The case therefore travels in the datum
itself, and no case files need to be written alongside. Neither the monitor
base class nor `filter_and_group` deduplicates by file path, so many datums can
all point at the same xlsx.

**Running and stopping.** Start with `courier run configs/courier/lsr-imagery.yaml`
and stop with Ctrl-C once the renderer has handled every case (the count is in
the logs and metrics). Courier keeps no seen-set across restarts, so after a
restart the sheet monitor emits every row again. Both scripts are idempotent
for that reason: `lsr_fetch.py` skips a case whose marker exists and whose row
hash is unchanged, and `lsr_render.py` skips a case whose outputs are newer than
its marker.

### Stage 1: sheet monitor (`lsr_sheet`, new plugin)

- Subclass `DataMonitorBasePlugin`. `find_file` polls the local xlsx (mtime,
  then a hash), reads "Seans Reports" with pandas, as
  `scripts/lsr_golden_to_jsonl.py` does, and keeps rows that have an
  `EVENT_ID` and a `Report Quality`. Only Sean's sheet, and only the local file.
  The formula cells are read from their cached values.
- For each new row, or a row whose content changed, yield:

  ```python
  Datum(
      file=xlsx_path,
      hostname="localhost",
      source="lsr_sheet",
      timestamp=report_time_utc,        # BEGIN_DATE_TIME_UTC rounded to the minute
      metadata={
          "event_id": "1309044", "case_number": 6558,
          "lat": 34.65, "lon": -87.79,
          "event": "Hail", "magnitude": 1.0,
          "radar": "KGWX", "radar_distance_km": 60.6,
          "quality": "Good", "notes": None,
          "row_hash": "…",
      },
  )
  ```

- Keep a seen-set in memory keyed on `event_id` + `row_hash`, so editing a row
  re-runs that case only, while the service is up.
- Config: `path`, `sheet` (default "Seans Reports"), `poll_interval_seconds`,
  and for the pilot runs `event_ids` and `limit`.
- The datum stays small. The storm history and the other reports for the
  overview map are read from the same xlsx by the scripts, by
  `case_number`, rather than packed into metadata.

### Stage 2: downloader (dispatcher + `scripts/lsr_fetch.py`)

The downloader is a dispatcher instance (`local_dispatcher`) with its own
name. Its payload does the downloading: courier dispatchers run payloads and
don't fetch anything themselves. For one case it:

1. Takes the report time from `{{ files[0].timestamp }}` and works out the
   window: −30 to +10 min. Hail falls after the peak, and the deck uses scans
   "a few minutes before".
2. Lists and downloads:
   - **Level II:** every volume from the closest radar inside the window (~6–8
     volumes, ~90 MB)
   - **MRMS:** for each product in the table above, every 2-minute file inside
     the window (~20 files × 14 products; most are small, composite and VIL are
     the large ones)
   - **SPC:** the hour at or before the report, for six parameters
3. Gunzips MRMS. Writes each file as `.part` and renames it when complete, so
   stage 3 never sees half a file.
4. Lays files out as `ready/<event_id>/{level2,mrms/<product>,spc}/...` and
   writes `ready/<event_id>/_COMPLETE_<YYYYMMDD-HHMM>.json` last. The name
   carries the report time. The content is the datum's metadata, the row hash,
   the file list, and anything missing (for example, a Level II gap).
5. Keeps a shared cache keyed by S3 key, using hard links, because
   reports in the same radar case share most files. 1,044 reports collapse to
   155 radar cases.

Exit non-zero on a failed download so courier's retry applies. Rate: keep the
downloader to a few concurrent jobs.

### Stage 3: ready monitor + renderer

- **Monitor.** This is the stock `file_system_poller_watchdog` on `ready/`,
  with a `data_monitor_config` plugin (`lsr_ready`, in the same style as
  courier's `goes16_abi`). It sets `source` and a `date` regex for each kind of
  file, so every datum gets its timestamp from the file name:

  | Entry | Match | source | date regex |
  |---|---|---|---|
  | case marker | `.*/_COMPLETE_\d{8}-\d{4}\.json$` | `lsr_case_ready` | `_COMPLETE_(?P<YYYY>\d{4})(?P<MM>\d{2})(?P<DD>\d{2})-(?P<HH>\d{2})(?P<NN>\d{2})` |
  | MRMS | `.*/MRMS_.*\.grib2$` | `mrms` | `_(?P<YYYY>\d{4})(?P<MM>\d{2})(?P<DD>\d{2})-(?P<HH>\d{2})(?P<NN>\d{2})\d{2}\.grib2` |
  | Level II | `.*/K[A-Z]{3}\d{8}_\d{6}_V06$` | `nexrad_l2` | `(?P<YYYY>\d{4})(?P<MM>\d{2})(?P<DD>\d{2})_(?P<HH>\d{2})(?P<NN>\d{2})` |
  | SPC | `.*/spc/\d{2}_[a-z]+\.gif$` | `spc_meso` | (from the case marker) |

  The monitor sees every downloaded, unzipped file. Only `lsr_case_ready`
  datums start a render, so rendering never begins with half a case's files.
  The per-file datums give log and metric visibility, and could later drive
  per-file jobs. That means only a config plugin is needed, not new monitor
  code.
- **Renderer** (`scripts/lsr_render.py <marker> --time <timestamp>`) calls
  GeoIPS from Python. It doesn't use `geoips run` per product, because sectors
  are made per case at run time (see below). It writes:

```
data/lsr-imagery/out/<event_id>/
  overview_refl_<time>.png                 # slide 1 style, reports plotted
  panel4_<time>.png                        # Z / V / ZDR / CC, slide 6 style
  top_divergence_<time>.png                # upper tilt Z + V, slide 7 style
  echo_heights.png + echo_heights.csv      # 50/60 dBZ heights over the window, melting level
  spc_<param>.png                          # cropped national SPC maps
  mrms/<product>_<time>.{png,nc,txt}       # LSR-AI tool equivalents; .txt in LSR-AI's table format
  case.json                                # report, quality, notes, file index
```

- `data/lsr-imagery/index.jsonl` lists every case and its outputs, for
  ReasonBench to load.

Not tracked by git, like `runs/`. Raw files in `ready/` can be deleted per case
after rendering, which keeps disk to a few GB at any time.

## GeoIPS work (plugin package `geoips_lsr`)

This is a separate GeoIPS plugin package in a new repository, `../geoips_lsr`,
next to this one in the P0089 project folder. It is registered with GeoIPS
through entry points, not a fork. It is built against GeoIPS
branch `dev-staging`, pinned to the commit current when work starts
(`1ea42ca9`, 13 Oct 2026, as of this plan) and moved forward deliberately.
GeoIPS on that branch requires Python 3.11 or 3.12. I checked that the pieces
the plan relies on are on `dev-staging`: the class-based reader interface, the
`gfs_grib` reader used as the model, the `center_coordinates` sector spec
generator, the `single_channel` algorithm, and the `imagery_annotated` and
`netcdf_geoips` output formatters.

Repository layout follows GeoIPS's plugin package template: `pyproject.toml`
with a `geoips.plugin_packages` entry point, `geoips_lsr/plugins/classes/`
for readers and output formatters, and `geoips_lsr/plugins/yaml/` for
products and colormaps. Tests go under `tests/`, with the two LSR-AI cases
as fixtures. Run `geoips config create-registries` after installing.
This repository (ReasonBench) keeps only the courier config and the
`lsr_fetch.py` and `lsr_render.py` scripts, and depends on the plugin package.

### Readers

1. **`mrms_grib2`**, modelled on `readers/gfs_grib.py` (pygrib):
   - 0.01° grid, 7000 × 3500, 20–55 N, 130–60 W. Crop to the area before
     building 2-D lat/lon; the full grid is 24.5 M points.
   - Mask −999 (no coverage), and keep −99 or 0 (no echo) as zero. The rubric's
     `missing_data_handling` criterion depends on telling these apart.
   - Metadata: `source_name: mrms`, `platform_name: mrms`, times from the GRIB
     validity time, one variable per product.
2. **`nexrad_level2`**, using xradar (or Py-ART):
   - One dataset per sweep, with gate lat/lon computed from range, azimuth and
     elevation. Variables: `reflectivity`, `velocity`,
     `differential_reflectivity`, `cross_correlation_ratio`. Attributes:
     elevation angle and beam height at the report's range.
   - Velocity: base velocity, dealiased (Py-ART region-based), and labelled as
     base velocity. Storm-relative velocity is a later option.
   - Storm-top tilt: pick the lowest elevation whose beam height at the storm
     is at or above `EchoTop_18` from MRMS.

### Products, colours, overlays

- Products YAML for each field. Colour tables: NWS standard for reflectivity
  (−30 to 80 dBZ), velocity (±125 kt), ZDR (−4 to 6 dB) and CC (0.45 to 1.0),
  to match the GR2Analyst look in the deck. MRMS products use fixed ranges
  (MESH 0–100 mm, VIL 0–80 kg m⁻², echo tops in kft).
- Sectors: the built-in `center_coordinates` sector spec generator, in Python,
  per case:
  - overview: centred between radar and report, about 450 × 250 km
  - panel: centred on the report, about 60 × 40 km
  - lsr_tool: the LSR-AI box, about 13 × 22 km at 1 km, taken from the
    extent of the tool tables in the logs
- Annotation: state and county lines (Census county shapefile; cartopy has no
  US counties), city labels (Natural Earth populated places), and markers for
  every report with the same `caseNumber`, read from the xlsx (green squares
  for hail, blue circles for wind, the case's own report highlighted). Markers need a small custom output
  formatter or a matplotlib step after `imagery_annotated`. GeoIPS has no
  point-marker overlay. A satellite basemap like GR2Analyst's is out of
  scope; plain background.
- 4-panel: compose the four single-product PNGs in the renderer (Pillow), not
  in GeoIPS.
- `lsr_table` output formatter: the cropped grid in LSR-AI's
  `latitude longitude value` text.

### SPC maps

Download only, with no GeoIPS. The national GIF is a fixed projection, so
cropping to a case region is a fixed lookup from lat/lon to pixels, fitted once
from the state outlines.

## Validation

1. **MRMS against LSR-AI.** For `0227test3` (2022-04-29 2348 UTC, 38.927 N,
   97.157 W), rebuild each tool table and compare it point for point with the
   tool results stored in `data/lsr-ai/0227test3.json` (MESH 7.4 mm at the
   nearest point, 47.5 dBZ at −20 °C, flag 7). This also shows which file
   LSR-AI picks for a time and whether it drops zero values. For `18cf24b6…`,
   find out whether its empty tables are an all-zero grid or a missing file.
   It also settles which VIL product LSR-AI uses: the tool is
   `get_vil_density` but the artifact key is `VIL_00.50`. Fetch both and keep
   the one that reproduces the log's VIL values (15.2 at the nearest point,
   9.5–20.9 within 1.5 km). Later phases fetch only that one.
   This becomes a pytest in the plugin package.
2. **Level II against the deck.** Run the two deck days (KFSD, 7 and 9 July
   2025) and put our 4-panels next to the deck's screenshots. The deck shows
   storm-relative velocity and we show base velocity, so only reflectivity, ZDR
   and CC should match closely. Then check the deck's stated numbers, for example "50 dBZ up to
   31,000 ft, 60 dBZ up to 20,000 ft" (slide 6) and "storm-top divergence of
   110 kt" (slide 10).
3. **Forecaster review.** Sean looks at about 10 golden cases, half Good and
   half Bad, including timing failures, and says whether he could have coded
   each one from our imagery alone.

## Phases

1. **Readers on known cases.** Build the MRMS and Level II readers and products,
   and run the renderer by hand on the two LSR-AI examples and the two deck
   days. Pass validation 1 and 2.
2. **Courier end to end on 20 reports** on a laptop: sheet monitor with an
   `event_ids` list, downloader, ready monitor, renderer. Check the downloader's
   size estimate and the retry behaviour.
3. **All 1,044 reports.** Move the renderer to `slurm_dispatcher` if needed.
   Expect roughly 155 radar cases × (~100 MB Level II + MRMS), so tens of GB
   downloaded in total, less after caching.
4. **Optional:** regional SPC maps rebuilt from RAP, storm-relative velocity,
   vertical cross-sections through the core (to show BWER), and GOES imagery
   (GeoIPS's `abi_netcdf` reader already exists).

## Decisions

- Base velocity for now; storm-relative velocity later, if at all.
- The image stamps in the deck are the correct dates (7 and 9 July 2025).
- Only "Seans Reports", read from the local xlsx.
- Courier runs as a service and is stopped with Ctrl-C; the scripts are
  idempotent so a restart is safe.
- Case data is passed to payloads through datum `timestamp` and `metadata`
  in the Jinja context.
- GeoIPS: branch `dev-staging`, pinned by commit. Plugin package: a new
  repository.

## Open questions

None remaining. The VIL product is settled empirically by validation 1.
