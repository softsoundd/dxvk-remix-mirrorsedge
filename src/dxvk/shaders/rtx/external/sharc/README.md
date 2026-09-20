# NVIDIA SHARC

Shader headers from [NVIDIA-RTX/SHARC](https://github.com/NVIDIA-RTX/SHARC) 1.8.3,
commit `4e21b585c33c83d723ca9a1e11bbb1090d145793`, under the NVIDIA RTX SDKs
license in `License.md`.

These headers carry local modifications, each marked `// NV-DXVK:` at the site:

- `HashGridCommon.h`, `HashGridTypes.h`, `SharcCommon.h` — ray portal space is part
  of the cache key. Crossing a portal rewrites the ray mask that feeds the
  continuation ray, the NEE shadow ray and the unordered resolve, so radiance at a
  point differs by the portal space that reached it and those cells must not share
  a key. Two bits are taken from the level field (9 to 7; levels are logarithmic in
  camera distance and do not approach 127) and the portal bits are carried across
  the level change so `SHARC_BLEND_ADJACENT_LEVELS` cannot mix portal space into
  main space. The reserved user-data bit 63 is untouched.

Remix-side adaptations of the SDK live outside this directory, in
`src/dxvk/shaders/rtx/pass/sharc/`.
