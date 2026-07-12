# EA Importance-Sampled FAST Noise

Vendored from https://github.com/electronicarts/importance-sampled-FAST-noise

License: BSD-3-Clause (see LICENSE.txt)

This directory contains 32 slices from EA's `FAST/` distribution:
`vector2_uniform_gauss1_0_Gauss10_separate05_<0..31>.png` — 128×128
two-channel noise with uniform spatial distribution and decorrelated
x/y channels under spatial filtering. Used by `RtxFastNoise` for cloud
ray-march jitter (see `src/dxvk/rtx_render/rtx_fast_noise.{h,cpp}`).

The "separate05" variant is what gives us the decorrelation property:
channel x is used for view-march jitter and channel y for sun-shadow
tap jitter, and the two channels do not co-vary spatially under DLSS
reconstruction.

Other distribution variants from the upstream repo (UniformStar,
UniformCircle, etc.) are intentionally not vendored — those are
shape-importance distributions designed for DOF lens kernels, not
generic 2D jitter.

To update: re-clone the upstream repo and re-copy the same 32 files.
The build script `scripts/embed_fast_noise.py` (created in Task 2) will
detect the change and regenerate `rtx_fast_noise_data.{h,cpp}`.

## Important: file ordering

EA's filenames use unpadded numeric suffixes (`_0.png` through `_31.png`),
so lexicographic sort gives `_0, _1, _10, _11, ..., _19, _2, _20, ...`
which is WRONG for slice ordering. The embed script must sort by the
integer extracted from the filename, NOT lexicographically.
