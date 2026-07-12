# Mirror's Edge fork + Remix Plus — maintenance guide

This fork combines **Mirror's Edge / UE3 compatibility** (`mirrors-edge` lineage) with the **Remix Plus** feature set. It is structured for pulling updates from both upstreams.

## Git remotes

| Remote | URL | Role |
|--------|-----|------|
| `origin` | `softsoundd/dxvk-remix-mirrorsedge` | This fork |
| `upstream` | `NVIDIAGameWorks/dxvk-remix` | NVIDIA runtime |
| `plus` | `RemixProjGroup/dxvk-remix` | Remix Plus features |

## Tags

- `me-baseline` — last ME-only commit before Plus integration (`97ba6557`)
- Create `me-plus-baseline` after integration completes for branching `mirrors-edge-plus`

## What lives where

| Area | Location | Merge strategy |
|------|----------|----------------|
| UE3 / Mirror's Edge | `src/d3d9/d3d9_rtx.cpp`, `src/d3d9/d3d9_device.cpp`, `dxso/`, `config.cpp` | **Never overwrite from Plus** |
| Plus features | `src/dxvk/rtx_render/rtx_fork_*.cpp` | Copy/cherry-pick from `plus/main` |
| Shared hooks | ~70 upstream files (see `docs/fork-touchpoints.md`) | Resolve at one-line `fork_hooks::` sites |
| ME atmosphere perf | `useSkyViewLut` in `atmosphere_args.h`, integrators | Keep when merging Plus atmosphere |

## Pulling NVIDIA upstream

```powershell
git fetch upstream
git merge upstream/main
```

Resolve conflicts using `docs/fork-touchpoints.md` — prefer keeping one-line hook dispatches and moving logic into `rtx_fork_*` modules.

## Pulling Remix Plus

```powershell
git fetch plus
git cherry-pick <plus-commit>   # preferred for focused fixes
# or
git merge plus/main           # larger releases; resolve in hook sites + rtx_fork_* only
```

Do **not** wholesale merge `plus/main` onto an old ME base — Plus may lag NVIDIA. Graft `rtx_fork_*` and post-sync commits onto the current ME+NVIDIA tip instead.

## Build & deploy (Mirror's Edge)

```powershell
meson compile -C _Comp64DebugOptimized
meson install -C _Comp64DebugOptimized --tags MirrorsEdge --only-changed
```

Bridge build required for 32-bit `d3d9.dll` in `Binaries/`.

See also [`docs/CONTRIBUTING.md`](CONTRIBUTING.md) (Remix Plus fork discipline).
