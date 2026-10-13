---
title: ArtemisThermalBase
order: 0
description: Illumination and regolith temperature for lunar polar terrain, on a laptop.
---

![Hottest surface temperature over one lunar day around Shackleton crater, from LOLA topography and the NAIF DE440 Sun](img/hero.png)

<p class="intro">ArtemisThermalBase traces sunlight across real lunar polar terrain and
follows the heat into the regolith underneath every facet. It tells you which
ground is lit, for how long, how hot and how cold it gets, and where water ice
could survive. <em>It runs on an ordinary laptop.</em></p>

<div class="spec">
  <div><b>79 202</b><span>facets of Shackleton at 200 m, one lunar day plus spin-up in 9 minutes on an eight-core laptop</span></div>
  <div><b>1.1 s</b><span>to build the ray-tracing tree for 583 000 triangles</span></div>
  <div><b>0.003°</b><span>agreement of our lunar orientation with SPICE MOON_ME</span></div>
  <div><b>1.5 GB</b><span>default memory ceiling; every run checks itself before it starts</span></div>
</div>

## What it computes

For each facet of a digital elevation model and each time step:

- the fraction of the solar disk it can see, with rays to many points of the
  disk so that penumbra and grazing light near the horizon come out right;
- the sunlight it absorbs, with an angle-dependent albedo;
- optionally the infrared it receives from surrounding terrain;
- the temperature of a 6 m regolith column under it, with depth- and
  temperature-dependent density, conductivity and heat capacity.

From these come maps of surface temperature, the hottest and coldest a place
gets, how much of the time it is in sunlight, and whether water ice there would
be stable against sublimation.

## Start here

1. [Getting started](guide.md): install, first run, real terrain.
2. [The desktop app](app.md): set up runs, inspect results, compare runs.
3. [Physics](physics.md): every equation the code solves, and why.
4. [Configuration](configuration.md): every setting, with its default.
5. [Outputs](outputs.md): what a run folder contains, for use in your own tools.

## What it is not

It is not validated against Diviner yet. The model, the data pipeline and the
numerics are tested, and the geometry agrees with NAIF SPICE, but a quantitative
comparison with LRO Diviner temperatures is still open. Read
[Validation and limitations](limitations.md) before you put a number from here
into a paper.

## Licence and citation

MIT licence. If the software helps your work, cite it with the metadata in
[CITATION.cff](../CITATION.cff), and cite the LOLA, NAIF and thermophysical
sources listed in [Data](data.md).
