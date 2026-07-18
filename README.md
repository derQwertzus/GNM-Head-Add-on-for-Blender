# GNM Head — Blender Add-on

Generate and sculpt parametric heads from Google's **GNM (Generative
Anthropometric Model)** directly inside Blender.

Semantic identity and expression sampling, full parameter sliders, painted
region masks for targeted randomization, clean native **quad topology**, and
one-click identity baking — driven by a small local server that the add-on
installs and manages for you.

**Author:** Alexander Börner

---

## Highlights

- **Semantic generation** — female↔male blend, ethnicity weight mixing, 20
  expression classes with intensity, independent seeds for head and
  expression. Everything applies live in the viewport.
- **Region-masked randomization** — six weight-painted masks (Eyes, Nose,
  Mouth, Jaw, Ears, Back of head) ship as ready-to-use defaults. With masks
  active, identity randomization only touches the masked areas; the viewport
  spotlights them (rest dims to 18%, specular off) so you always see what
  you're editing. Paint your own masks and save them once — every future
  head loads them automatically.
- **Native quad mesh** — the model's authored edge loops, not a
  triangulation. Welded vertices with per-loop UVs (seams live only in UV
  space). Vertex colors included as a point attribute (grayscale clay by
  default, configurable).
- **Component toggles** — show/hide eyes, teeth, tongue, and the cornea
  shells; hidden components are fully removed (faces *and* vertices).
- **Full parameter access** — Identity Advanced, Expression Advanced, and
  Head Pose panels expose every model component with real component names.
- **Bake Identity** — copies the current head as an independent, cleanly
  colored object; generate on, bake variants as you go.
- **Lightweight by design** — semantic sampling runs on a bundled **NumPy
  backend** (h5py reads the decoder models directly), so **no TensorFlow**
  is required. The whole managed environment is **~170 MB**, most of it the
  model data itself.

## Requirements

- **Blender 4.2+** (extension format; developed on 5.x)
- A system **Python 3.10–3.12** on PATH (used once, to create the managed
  environment)
- Internet access for the one-time environment install

## Installation

1. Grab the extension zip from **Releases** (or build it yourself: zip the
   repo contents so `blender_manifest.toml` sits at the zip root, or run
   `blender --command extension build`).
2. In Blender: `Edit → Preferences → Get Extensions → ⌄ → Install from
   Disk…` and pick the zip.
3. Open the add-on's preferences and press **Install Environment**. The
   add-on creates a private virtual environment and downloads everything
   from original sources — GNM from
   [github.com/google/GNM](https://github.com/google/GNM), the small Python
   dependencies from PyPI. Progress is shown live; afterwards the panel
   lists the measured size (~170 MB) and the exact install locations.
4. `3D Viewport → N-panel → GNM Head → Generate Head`. The managed server
   starts automatically (first start loads the model, ~10–30 s) — by
   default in a visible console window with the live log.

An external GNM server works too: point **Server URL** at it and skip the
managed install.

## Panels

| Panel | What it does |
|---|---|
| **GNM Server** | Server status, start/stop, console toggle, test connection, refresh/reconnect |
| **Generate** | Seeds + dice buttons, gender/ethnicity weights, expression + intensity, component toggles |
| **Mask Regions** | Mask visualization toggle, one button per region, save/create mask groups, clear masked changes |
| **Identity / Expression Advanced, Head Pose** | Every model parameter as a slider, grouped, with show all / show less |
| *(main)* | Reset All and the **Bake Identity** button |

## Notes

- The UI is active only while the GNM head is the selected, visible object —
  no invisible background updates.
- Masked identity changes are a vertex-space blend held in memory; they
  survive slider edits but not a Blender restart. **Bake identities you want
  to keep.**
- Uninstalling the environment (preferences) removes the entire managed
  folder: environment, sources, logs, and saved masks.
- Skin color and saturation of the vertex colors are configurable in
  `server/config.py`.

## Repository layout

```
blender_manifest.toml        extension manifest (author: maintainer field)
__init__.py                  the add-on
default_region_masks.json    bundled region masks (full-index, painted)
server/server.py             bundled GNM server (Flask)
server/config.py             server settings: port, skin color, saturation
server/numpy_sampler.py      NumPy inference for the semantic decoders
server/verify_semantic.py    optional: compare NumPy vs TensorFlow backends
```

## License

- Add-on: **GPL-3.0-or-later** (Blender extension requirement)
- GNM model & sources: **Apache-2.0**, © Google — downloaded from the
  official repository at install time, not redistributed here.
