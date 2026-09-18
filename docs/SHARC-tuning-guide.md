# Tuning SHARC — a practical guide

For people running Remix on a game: you edit `rtx.conf`, you can read a stats panel, and you
want the cache to actually do something. No renderer background assumed.

SHARC is a **radiance cache**. While the path tracer works, it also writes what it learns about
the lighting at each point it hits into a grid of cells spread through the world. Later, when a
path lands somewhere a cell already knows about, the path stops there and reads the answer out
of the cell instead of tracing further. That buys lower noise, steadier lighting in motion, and
cheap extra bounce depth. It costs a little blur, because a cell is one average over a region of
space and over several frames.

**On measurement.** Everything numeric here is labelled. **Measured** means somebody read it off
a running game and it is recorded in this repo's notes — that is Portal RTX and Fallout New Vegas
open desert, and nothing else. **Expected** means it follows from the code but nobody has looked.
FNV *interiors*, which are most of that game, have never been sampled. Where the honest answer is
"nobody knows", it says so.

---

## 1. Turn it on, and a starting configuration

```ini
rtx.integrateIndirectMode = 3
```

`3` is SHARC. (`0` plain path tracing, `1` ReSTIR GI, `2` NRC — the stock default.)

**In most cases you do not need to paste anything.** The shipped defaults are the Balanced
preset, which is the profile below, so `rtx.integrateIndirectMode = 3` on its own gets you there.
The panel also has a **SHARC preset** dropdown with Quality, Balanced and Performance. Only five
of the fourteen values a preset writes differ between them — the update tile size, the update
bounce count, the sky retries, the primary-vertex deposit and the cache capacity. The other nine
are correctness and coverage controls that all three write identically.

Paste this only if you want the settings written out explicitly, or you are on an older build
whose defaults predate the presets:

```ini
rtx.integrateIndirectMode = 3

# The line that decides whether the cache does anything at all. Do not skip it.
rtx.sharc.allowSpecularPaths = True

# Quality gates
rtx.sharc.footprintGate       = True    # default; leave on
rtx.sharc.minRoughness        = 0.05
rtx.sharc.maxEmissiveLuminance = 0.1
rtx.sharc.minSampleCount      = 2       # default

# Budget
rtx.sharc.capacityLog2     = 20
rtx.sharc.updateTileSize   = 8
rtx.sharc.updateBounces    = 4
rtx.sharc.accumulationFrames = 4
rtx.sharc.staleFrames      = 32         # default
rtx.sharc.gridScale        = 50         # default
```

Why these values, and what each one is protecting you from:

| Option | Value | Why |
|---|---|---|
| `allowSpecularPaths` | `True` | Off, the cache refuses every surface a non-diffuse ray reached. In an enclosed scene that is nearly everything — **measured**: Portal RTX sat at 0.2% eligible surfaces purely because this line was missing from its config. |
| `minRoughness` | `0.05` | A *squared* roughness, so it is stricter than it looks. With `footprintGate` on, the old reason for keeping it high is gone. |
| `maxEmissiveLuminance` | `0.1` | At 0 *any* emission at all disqualifies a surface, which throws out every faint emissive map. **Measured** in Portal: this alone took the emissive reject share from "almost every surface" down to 0.8%. |
| `minSampleCount` | `2` | Stops a cell answering from a single sample. **Measured**: this is what removed the glow on geometry coming into view. |
| `capacityLog2` | `22` | Occupancy was never measurably short even at 20, but the resolve pass runs one thread per slot, so this costs a little every frame regardless. See §5. |
| `updateTileSize` | `8` | One update path per 8×8 pixel tile. **Measured** Portal still hit 99.2% at this rate. |
| `updateBounces` | `4` | 8 was more depth than the cache needed. Note this is *not* `rtx.pathMaxBounces`. |
| `accumulationFrames` | `8` | Response to lighting changes versus per-cell noise. See §4 before changing it in either direction. |

Two more are **on by default**: `rtx.sharc.updatePrimaryVertex = True` and
`rtx.sharc.updateSkyRetries = 1`. They recover update work that would otherwise be thrown away by
paths escaping to the sky, and the retry half matters only where the sky is visible. **Measured**
(user report): both improve quality, and the primary deposit costs **about 0.1 ms** — the same
order as the whole cache's net benefit, so treat SHARC as a quality feature with a roughly neutral
frame-time story rather than a performance one. The scene each figure was read in is not recorded.
See §5.

One is **off by default and is the answer to a specific complaint**: `rtx.sharc.maxDepositLuminance`.
If `allowSpecularPaths` is giving you **fireflies on reflective materials** — and especially if they
get worse the lower you set the DLSS preset — that option is the cheap cure, and it is the only one
that does not cost you cache coverage. It needs a number picked per scene from debug view 583, so it
is not in the config block above. See §5 and
[SHARC-specular-fireflies-2026-09-16.md](SHARC-specular-fireflies-2026-09-16.md). **Unmeasured.**

**Requirements.** SHARC needs shader Int64, buffer Int64 atomics, FP16, 16-bit storage and
RayQuery. On a device missing any of them it **silently traces ordinary paths** — no error, no
visual difference from mode 0, and the panel says "Unsupported device features".

---

## 2. Reading the stats panel

Turn on both of these — the stats option does nothing without the timing one:

```ini
rtx.sharc.measureGpuTime    = True
rtx.sharc.collectQueryStats = True
```

The panel lives in the SHARC section of the Remix settings UI. It samples 1 in 64 pixels and
sums over a **120-frame window**, so nothing appears until roughly two seconds after you enable
it, and the numbers then refresh every two seconds. Let the "Cache age" line pass ~120 frames
after any change that clears the cache before you believe a reading.

> **Turn both off before you time anything.** **Measured**, FNV: the query pass read 8.82 ms with
> statistics off and 9.78 ms with them on. Call it a millisecond. Timing boundaries also break up
> GPU overlap, so `measureGpuTime` distorts total frame time on its own.

### The lines, and which one to act on

**`Cache terminates X% of paths | Y segments/path`**
The share of paths that ended on a cache read instead of tracing onward, and the mean number of
traced ray segments per path. This is the payoff line — it is what the cache is actually buying.
**Measured**: 78.5% in Portal RTX, 23.3% in FNV open desert.

**`Lookup hit rate X% | eligible surfaces Y%`**

- *Eligible surfaces* is the share of surfaces the path hit that the cache is even allowed to use.
  **This is the first number to look at.** If it is near zero the cache is gated off, not
  underperforming, and no amount of tuning the budget settings will help. See §3.
- *Lookup hit rate* is, of the lookups actually attempted, how many came back with a usable cell.
  It is a **diagnostic, not a target** — see §4.

**`Of misses: no cell X% | below sample floor Y%`**
This splits every miss into the two failures that have *opposite* fixes:

- **no cell** — nothing has ever been stored at that spot, or it was stored and then evicted.
  The fix is population: more update samples (`updateTileSize` down), longer retention
  (`staleFrames` up), fewer/larger cells (`gridScale` down), or more room (`capacityLog2` up).
- **below sample floor** — a cell exists but has not collected enough samples to be trusted yet.
  The fix is trust and feeding rate: `accumulationFrames` up, `minSampleCount` down, and again
  `updateTileSize` down.

Getting these two backwards wastes hours. Read the split before you touch anything.

**`Surface rejects: roughness X% | incoming non-diffuse Y% | other Z%`**
Why ineligible surfaces were refused, as a share of all resolved surfaces.

- *incoming non-diffuse* large → `allowSpecularPaths` is off. This is the classic dead-cache
  cause.
- *roughness* large → surfaces are smoother than `minRoughness` allows. Lower `minRoughness`
  (floor is 0.05). If it is still large at 0.05, that is the honest quality wall, not a bug:
  **measured** Portal still rejects 15.8% on roughness at 0.05, because its panels are shinier
  than an isotropic cache can stand in for.
- *other* is broken out on the next line.

**`Other: non-opaque | medium | opacity < 1 | subsurface | emissive`**
The five surface terms behind "other". *emissive* large → raise `maxEmissiveLuminance`.
*opacity < 1* large → foliage and grate cutouts, nothing to do. *non-opaque* is glass.
**Measured** Portal at the recommended profile: medium 5.8%, non-opaque 3.9%, emissive 0.8%.

**`Too close: X% of eligible surfaces | of those: last leg only | post-portal | first bounce`**
A lookup is refused when the ray segment that arrived was shorter than the cell's own diagonal,
because the path might still be inside the cell it is about to read — that would be the path
reading its own answer back. The threshold is a fixed *fraction of camera distance*
(about 1.7–3.5% of it at `gridScale` 50), so it is centimetres in a room and metres across a
desert. **Measured**: 4.1% of eligible in Portal, 19–30% in FNV open desert. Indoors this number
is not worth chasing. The sub-splits are instrumentation for a possible future change; ignore
them unless *last leg only* is large, which would mean the guard is being tripped by re-traces
through cutouts rather than by genuinely short segments.

**`Footprint too narrow: X% of eligible surfaces`**
Specular arrivals refused by `footprintGate` — the reflected lobe was still tighter than a cell
when it landed, so serving it a cell average would have made it glow. Expected to be nonzero and
healthy. This is coverage you are giving up on purpose.

**`Path ends: sky | bounce limit | zero weight | roulette`**
How paths finished when the cache did *not* end them. **The sky number is the ceiling on
everything else.** A path whose first bounce leaves for the sky never consults the cache at all,
so cache terminations can never exceed `(1 − sky share) × hit rate`. **Measured** FNV open
desert: sky 65.9%, which caps terminations near 34% however good the cache gets. **Measured**
Portal, enclosed: roulette 21.0%, sky negligible.

**`GPU ms: update | resolve | query`**
The three SHARC passes. *update* is the sparse pass that fills the cache, *resolve* averages
every cell once per frame, *query* is the full-resolution indirect pass. When you change
`updateTileSize` or `updateBounces`, *update* is the line that should move. When you change
`capacityLog2`, *resolve* is.

**`Cache age: N frames`** — frames since the cache was last cleared. If it keeps resetting to 0,
something is changing a setting that clears it (see §5) or the renderer keeps resetting history.

**`Fell back on N of M frames`** — frames where SHARC was selected but did not run. See §3.

---

## 3. Symptom → fix

### The cache appears to do nothing. Eligible surfaces near zero.

**By far the most common failure, and it is almost always config, not tuning.** Work down this
list in order:

1. **`rtx.sharc.allowSpecularPaths` is missing from that game's `rtx.conf`.** It defaults to
   `False`, and with it off only surfaces reached by a *diffuse* bounce are eligible. In an
   enclosed space where every wall is reached off some other surface's glossy lobe, that is
   effectively nothing. **Measured**: this single missing line held Portal RTX at 0.2% eligible.
   Turn on debug view **581 (Rejection Reason)** — if the screen is flooded cyan, this is it.
2. **Check the panel's fallback line.** SHARC can be selected and still not run:
   - *"Unsupported device features"* — the GPU lacks Int64 atomics / FP16 / RayQuery.
   - *"Raytraced render target active"* — the game is drawing a ray-traced in-world screen this
     frame. SHARC drops out for the whole frame. Governed by `rtx.raytracedRenderTarget.enable`.
   - *"Ray portals block SHARC"* — set `rtx.sharc.allowRayPortals = True`.
   - *"Cache allocation failed"* — lower `capacityLog2`, then press Reset SHARC.
3. **`minRoughness` too high.** Older builds defaulted to 0.8, which is strict, and it is a *squared* roughness (§4).
   Drop it to 0.05 and watch the roughness reject share.
4. **`maxEmissiveLuminance` at 0.** Every surface carrying even a faint emissive map is refused.
   **Measured** in Portal: at 0.1 the emissive rejects fell to 0.8% of surfaces.
5. **You are on an older build.** Graphics presets used to move `integrateIndirectMode` off SHARC
   when clicked. That is fixed in this tree — the preset now explicitly leaves an explicit SHARC
   selection alone — but on an older DLL, clicking a preset silently switched you to NRC.

### Reflective surfaces glow or blow out

The cache stores one average that points in every direction. A shiny surface needs a *directional*
answer, so handing it the average makes it too bright.

- Confirm `rtx.sharc.footprintGate = True`. This is the real fix and it is on by default. It
  measures how wide the reflected lobe has spread by the time it reaches the surface and refuses
  the lookup if the lobe is still tighter than a cell. **Measured** (user report, Portal): a large
  improvement, with far more usable data in the cache than the old roughness-threshold approach,
  even with `minRoughnessSpecular` down at 0.05.
- If it persists, raise `minRoughness` in steps of 0.05–0.1 until reflections stop glowing, then
  back off one step. Expect to lose coverage.
- Do **not** reach for `minRoughnessSpecular` — with `footprintGate` on it does nothing at all
  (§5), and its slider is hidden for that reason.

### Glow on camera movement, especially on geometry that was just off screen

Cells for off-screen geometry never get updated. The first path to land in one, the moment it
comes into view, can be read back as if it were a converged answer — and one path is a very
noisy answer.

- **`rtx.sharc.minSampleCount = 2`** is the fix and is now the default. **Measured** (user report,
  Portal): it removed this glow, which the roughness thresholds had only ever masked. The SDK's
  own default is 0, i.e. one sample is enough — do not go back there.
- If you are running `minSampleCount = 0` or `1`, that is your answer.

### Hit rate collapses outdoors

Expected, and only partly fixable. Two things are happening.

1. Most update paths launched from open ground exit to the sky on their first bounce and store
   nothing. **Measured** FNV: 65.9% of paths end at the sky, so two thirds of the update budget
   buys no cache sample at all.
2. A cell outdoors is fed only by the few nearby rocks and walls whose bounce rays reach it, not
   by a whole room's worth of surfaces. The feeding rate per cell is roughly 5–30× lower than
   indoors (**expected**, from the code — never directly measured).

Act on the miss split:

- **below sample floor dominant**: raise `accumulationFrames` to 32 (then 64 if needed). A cell
  fed once every *k* frames is only ever readable if *k* is smaller than `accumulationFrames`, so
  at 4 anything fed less than every fourth frame is permanently stuck below the floor. Then
  `staleFrames` to 128, but *only together with the accumulation raise* — alone it just keeps
  cells whose history the next sample destroys. Then `updateTileSize` 8 → 4 for 4× the samples,
  at 4× the update cost.
- **no cell dominant**: try `capacityLog2 = 22` for one session. If the share does not move,
  capacity is exonerated — go back to 20 and treat it as a density problem instead.
- Two options exist specifically for this and are **on by default**:
  `updatePrimaryVertex` (every path stores something, even sky-bound ones) and `updateSkyRetries`
  (re-aim a first bounce that missed). Both are confirmed to improve quality (user report); the
  primary deposit costs about 0.1 ms. See §5.

**But read `Path ends: sky` first.** If it says 66%, terminations are capped near 34% no matter
what you do, and the cache's product outdoors is *stability*, not speed. Do not spend interior
quality to chase an outdoor hit rate that buys nothing.

### Lighting lags behind changes — a light switches, a flash goes off, the indirect catches up late

`accumulationFrames` is a time constant in frames. At 32 the indirect channel trails a lighting
change by roughly half a second at 60 fps. Lower it — 4 to 8 is responsive — and accept a noisier
cache. If you raised it to fix an outdoor hit rate, this is the bill for that.

Note the split: the direct light changes instantly and only the indirect bounce lags, so it reads
as the bounce light "arriving late" rather than the scene being slow.

### The cache keeps resetting (Cache age stuck near 0)

Most quality settings clear the cache when changed, which is correct but makes live slider-dragging
useless. Set the value, let go, and wait ~120 frames. If the age never climbs while you are not
touching anything, something else is resetting renderer history — a camera cut, a scene change, or
a setting being written every frame by a preset.

### Everything looks right, but SHARC looks identical to plain path tracing

That is the expected result, and it is not a failure. With zero cache hits the SHARC query pass
runs the *same* integrator as `rtx.integrateIndirectMode = 0` at every site that can change a
pixel. With hits, the cache replaces the path's tail with an average of samples of the same
estimator, so the expected brightness does not change — only its variance. "Identical but steadier
in motion" is the signature of a correct cache applied to shallow light transport.

One caveat: there is exactly one site where SHARC and mode 0 can differ, and it only matters if
`rtx.di.enableSampleStealing` is on. DLSS Ray Reconstruction's path-tracer preset turns that
option off, so under DLSS-RR the two really are the same on a miss.

---

## 4. Pitfalls that cost real time

**Hit rate is a diagnostic, not a target.** A refused lookup counts as a miss. The `too close`
guard and the `footprintGate` both *deliberately* refuse lookups that would look wrong, so turning
on a safety feature can lower your hit rate while improving the image. Chase `Cache terminates`
and the picture, not the hit rate. When `Cache terminates` stops moving, stop tuning.

**`accumulationFrames` and `minSampleCount` close the same pipe from opposite ends.** A cell is
readable only when it holds more than `minSampleCount` samples, and its sample count is repeatedly
scaled down as `accumulationFrames` elapses. A cell fed once every *k* frames settles at
`1 + accumulationFrames/k` samples — so at the starting profile above (`accumulationFrames` 4,
`minSampleCount` 2) a cell fed every 4th frame converges to exactly 2.0 and *never* passes the
strictly-greater test. It exists, it is being fed, and it is permanently unreadable. Set both
aggressively and you can strand the cache: cells full of data that nothing is ever allowed to
read. If "below sample floor" is large, raise accumulation *before* you lower the floor.

**Lowering `minRoughness` alone used to make smooth surfaces glow. `footprintGate` is the actual
fix.** The old approach held specular arrivals to a separate, higher roughness threshold
(`minRoughnessSpecular`), which was a blunt proxy — it refused *every* specular arrival at a
surface below the floor regardless of how wide the arriving reflection actually was. The footprint
gate measures the lobe instead, so narrow reflections are refused and broad ones are admitted on
their merits. It is on by default. If you inherited a config with `minRoughnessSpecular` cranked
up, you can drop it; it is inert anyway.

**`minRoughness` is a *squared* roughness.** 0.05 squared-roughness is roughly 0.22 in the
perceptual roughness a material editor shows you; 0.8 squared is roughly 0.89 perceptual.
So 0.05 is far less permissive than the number suggests, and 0.8 is nearly "matte only". The
in-game slider says "(squared)" for this reason.

**Interiors and open skies are different regimes, and settings do not transfer.** Enclosed
geometry is where SHARC pays: every wall illuminates every other, cells are fed every frame, paths
have a long tail for a cache hit to cut. Outdoors most paths exit to the sky at the first bounce,
never consult the cache, and cannot be helped by it — and the first bounce is never saved either
way. There is a hard ceiling out there, and it is printed on the panel as `Path ends: sky`. Tune
for whichever your game mostly is, and let the other regime fall back to plain path tracing, which
is what it does — **measured** cost in FNV desert: about 0.1 ms for a cache that does nothing.

**`gridScale` is angular and unit-free.** It does not need calibrating per game. A cell's edge is
the vertex's distance from the camera (rounded down to a power of two) divided by `gridScale`, so
the cell size is an *angle*, not a length — about 0.57° to 1.15° of arc at 50, anywhere in any
scene. Scale a whole world by any factor and the cell structure is identical. That means a tuning
result carried from one game to another carries cleanly, and it means `rtx.sceneScale` is
irrelevant here (coupling them would actively break it). Also note the direction: **larger values
mean *finer* cells**, and quadratically more of them.

**The stats and timing options are not free.** About 1 ms with statistics on (**measured**), plus
whatever the timestamp boundaries cost in lost GPU overlap. Turn both off before any A/B where the
number you care about is frame time.

---

## 5. Every setting

Each entry: what it does, when to move it, which way, what too far looks like, and — the part
that costs people hours — **what it depends on**. Several of these do nothing, or do something
different, depending on state elsewhere.

### Eligibility — which surfaces the cache may use

#### `rtx.sharc.allowSpecularPaths` — default `True`

Lets the cache store at and read from rough opaque surfaces that a *non-diffuse* ray arrived at.
Off, eligibility collapses to "the arriving ray came from a diffuse bounce", which in an enclosed
space is almost nothing.

**Turn it on.** This is the master switch for the entire specular half of the feature and its
absence is the single failure that made the cache look dead in Portal RTX. Too far does not really
exist for this one; the gates below are what control the quality cost.

> **Depends on / affects:** nothing gates *it*, but it gates plenty. With it **off**,
> `footprintGate` and `minRoughnessSpecular` both do nothing at all (they only ever apply to
> specular arrivals), and their controls are hidden in the UI. Changing it clears the cache.

#### `rtx.sharc.footprintGate` — default `True`

The quality gate for specular arrivals. Measures how far the reflected lobe has spread by the time
it reaches the surface — `segment length × sqrt(0.5·α²/(1−α²))`, α being the roughness of the
surface that *launched* the ray — and refuses the lookup unless that footprint exceeds the cell
size. This is NVIDIA's prescribed test.

**Leave it on.** Turn it off only to A/B against the old behaviour. Off, smooth surfaces get served
cell averages and glow.

> **Depends on:** `allowSpecularPaths` must be on or this is inert. While it is **on**,
> `minRoughnessSpecular` is not used at all and specular arrivals share `minRoughness` with diffuse
> ones. It never applies to diffuse arrivals — a diffuse lobe is as wide as lobes get. Changing it
> clears the cache.

#### `rtx.sharc.minRoughness` — default `0.05`, range 0.05–1.0

The roughness floor for caching a surface at all. Below it, the surface is considered too shiny for
an isotropic average to stand in for its reflection.

**Lower it** to widen coverage — 0.05 is the floor and is a reasonable place to sit with the
footprint gate on. **Too far** looks like reflections flattening out and losing their sense of
direction, or gaining a haze. Back off one step when you see that.

Remember it is *squared* roughness — 0.05 here is about 0.22 perceptual.

> **Depends on:** nothing, and it applies to every arrival. Note the host **clamps** it to
> [0.05, 1.0]; a config asking for less gets 0.05. (There was a bug where it was silently clamped
> to 0.5 while the slider showed your value — fixed in `7dcd68206`. If you are on an older DLL,
> every experiment you ran below 0.5 measured nothing.) `minRoughnessSpecular` is additionally
> floored at whatever this is. Changing it clears the cache.

#### `rtx.sharc.minRoughnessSpecular` — default `0.7`

The old, stricter roughness floor applied only to surfaces a specular ray arrived at.

**Almost certainly do not touch this.** It is kept so the pre-footprint-gate behaviour can still be
A/B'd.

> **Depends on:** **inert whenever `footprintGate` is on** — which is the default. Its slider is
> hidden in that case, so if you are setting it in `rtx.conf` and seeing no change, this is why.
> It also requires `allowSpecularPaths` on. And the host floors it at `minRoughness`, so setting it
> *below* `minRoughness` does nothing. Changing it clears the cache.

#### `rtx.sharc.maxEmissiveLuminance` — default `0.1`

The cache stores *reflected* light, and the path adds a surface's own emission separately, so
emissive surfaces are excluded to avoid confusion. At the default 0, any emission whatsoever
disqualifies a surface — which rejects every faint emissive map, and in a game that uses them
liberally that is most of the level.

**Raise it** until the emissive reject share on the panel drops to something small; 0.1 is a good
start. **Too far** is emissive surfaces bleeding their own glow into the cache and out onto their
neighbours.

> **Depends on:** nothing. Applies identically to storing and reading. Changing it clears the cache.

### Trust — when a cell is allowed to answer

#### `rtx.sharc.minSampleCount` — default `2`, range 0–32

A cell is ignored until it holds *more* than this many accumulated samples. At 0 — which is the
SDK's own default — a single path's result can be read back as though it were converged, which is
exactly what makes newly-revealed geometry glow when you turn the camera.

**Leave at 2.** Lower it only to buy coverage in a scene where cells are genuinely starved, and
expect the glow back. Raise it if you still see two-sample noise, at the cost of freshly revealed
areas staying uncached longer.

> **Depends on:** couples hard with `accumulationFrames` — see §4. The two together decide whether
> a sparsely-fed cell is *ever* readable: at `minSampleCount` 2, a cell fed every *k* frames is
> readable only when *k* < `accumulationFrames`. Changing it clears the cache.

#### `rtx.sharc.accumulationFrames` — default `8`, range 1–64

How many frames of samples a cell blends together. Also the cache's response time to lighting
changes, in frames.

**Lower** (4) for responsiveness in a scene where cells are fed every frame — any enclosed space.
**Raise** (32, then 64) when the panel says "below sample floor" dominates the misses, which means
cells are being fed too slowly to ever clear the floor. **Too far up** looks like indirect light
trailing half a second behind a light switch, a muzzle flash or an explosion.

> **Depends on:** gates against `minSampleCount` (§4). **Takes effect live — does not clear the
> cache**, so you can drag this one and watch.

#### `rtx.sharc.staleFrames` — default `32`, range 8–128

How long a cell survives without receiving a sample before it is evicted, key and all. At 32 and
60 fps that is about half a second — turn away from a wall for longer than that and its cells are
gone.

**Raise it** (to 128) when "no cell" dominates the misses and you pan a lot. But raise
`accumulationFrames` with it: on its own it just preserves cells whose accumulated history the
next sample immediately crushes, moving them from "no cell" to "below sample floor" without making
them readable.

> **Depends on:** pointless without a matching `accumulationFrames`. **Takes effect live — does not
> clear the cache.**

#### `rtx.sharc.maxDepositLuminance` — default `0` (off). **Unmeasured.**

Caps the luminance of a single value an update path writes into a cell. The cure for **fireflies on
reflective materials** with `allowSpecularPaths` on.

A cell is a mean, so one outlier is never averaged away — only divided by the cell's sample count,
which works out as `L / ((accumulationFrames + 1) * k)` for a cell fed `k` times a frame. And `k`
falls with the **render** resolution, because the update pass traces one path per `updateTileSize`
tile of the render target. That is the whole explanation for the two things people notice: fireflies
get worse the lower the DLSS preset (about **9x** worse at Ultra Performance than at DLAA, at 1440p),
and lowering `updateTileSize` cures them (4 quarters them). Both move the same divisor, and only one
of them is free.

**Set it** from debug view **583 (Cached Radiance)**: read the brightest cached radiance you
legitimately want, then set this comfortably *above* it — 583 shows the cell mean, and this bounds a
single deposit, which is larger. **Too far down** looks like bright cached areas going flat before
the fireflies go.

Unlike every other remedy it **costs no coverage**: it refuses no lookup, rejects no surface and
loses no cell, so it cannot take back the detail `allowSpecularPaths` buys. What it costs instead is
bias — a cell whose true radiance is above the threshold is stored dark.

> **Depends on:** `rtx.sharc.deferredUpdates` must be on (it is by default); the comparison backend
> ignores this. **Takes effect live — does not clear the cache**; old values wash out in
> `accumulationFrames` frames, so you can drag it and watch. Full reasoning, with the arithmetic and
> the four alternatives that were ranked below it:
> [SHARC-specular-fireflies-2026-09-16.md](SHARC-specular-fireflies-2026-09-16.md).

### Budget — what the cache costs

#### `rtx.sharc.capacityLog2` — default `22`, range 18–22

Cache size as a power of two. 20 is 1M cells / 40 MiB, 21 is 80 MiB, 22 is 160 MiB. When all 16
slots of a hash bucket are taken, the new cell is silently dropped — no counter records it, which
is why the only way to test capacity is to change it and see if anything moves.

**Raise it** as a one-session experiment when "no cell" dominates. If the share does not move,
capacity was not the problem — go back down. **Expected** (never measured) load at 20 in a wide FNV
desert vista is 0.1–0.3, i.e. comfortable. Raising it costs memory and makes the resolve pass run
more threads every frame.

> **Depends on:** nothing. Reallocates buffers and clears the cache on change. Watch
> `GPU ms: resolve` when you raise it.

#### `rtx.sharc.updateTileSize` — default `8`, range 1–16

One cache-filling path is traced per N×N pixel tile. 8 means 1/64 of the pixels. This is the main
lever on how densely the cache is fed.

**Lower it** (8 → 4) for 4× the samples per cell when cells are starved; **raise it** when the hit
rate is already in the high 90s and you want the update pass cheaper. **Measured** Portal at tile 8:
99.2% hit rate — over-served, and the obvious place to claw back cost. **Too far down** shows up
directly on `GPU ms: update`; indoors those update paths are the long ones, so the cost climbs fast.

> **Depends on:** nothing. **Takes effect live — does not clear the cache.**

#### `rtx.sharc.updateBounces` — default `4`, range 1–8

How many bounces a cache-filling path runs. Russian roulette is forced off for these paths, so they
run the full count unless they miss or lose all their weight first.

**Lower it** (4) to cut update cost. **Raise it** when the panel shows `Cache terminates` well below
the hit rate indoors — cells exist, but the update paths are stopping short of where queries land.
Outdoors it is nearly moot: update paths end at the sky anyway.

> **Depends on:** this is **not** `rtx.pathMaxBounces`. Update paths ignore the global bounce limit
> entirely and use this instead; query paths use the global one. Also: with `updatePrimaryVertex`
> on, the primary takes a slot, so at `updateBounces = 8` the deepest vertex no longer gets its own
> cell (its light is credited to the previous one, which is correct but is one less cell). Changing
> it clears the cache.

#### `rtx.sharc.gridScale` — default `50`, range 1–1000 (NVIDIA documents 1–100)

Cell density. **Larger = finer cells**, quadratically more of them. It is an angle, not a length:
at 50 a cell spans 0.57°–1.15° of arc at any distance in any game, which is roughly 16–32 pixels
at 2560px across a 90° field of view.

**Lower it** (50 → 25) to get four times fewer, four times better-fed cells when cells are starved.
**Too far down** means indirect lighting averaging across features it should not — an inside corner,
the shadow under a table. Raising it does the reverse and multiplies cell count fast.

> **Depends on:** nothing, and specifically **not `rtx.sceneScale`** — the grid normalises world
> units away by construction, so one value suits every game. Note it moves two other things with it:
> the `too close` threshold and the footprint gate both scale with cell size, so lowering `gridScale`
> makes both *stricter* (they demand longer segments / wider lobes). That cost lands hardest at long
> view distances, where the threshold is already metres. Changing it clears the cache.

### Compatibility and the opt-in experiments

#### `rtx.sharc.allowRayPortals` — default `False`

Lets SHARC run while a ray portal is actually open.

**Turn it on in a portal game.** **Measured**: a 21-minute Portal RTX session with portals active
and zero fallbacks, and the user reported portal content looked better than with the other indirect
samplers.

> **Depends on:** only matters when a portal is genuinely open this frame — merely having portal
> texture hashes configured does not trigger the gate (an earlier version tested for that and
> disabled SHARC for entire sessions in any game that defines portals at all). Portal-reached
> vertices get their **own cells**: portal space is part of the cell key, so they cannot contaminate
> what main-space paths read. Changing it clears the cache.
>
> *Note: the option's own description text and its in-UI tooltip are stale — they still say portal
> vertices are never inserted and that the feature is untested. Both statements predate the
> portal-space key change and the Portal RTX session.*

#### `rtx.sharc.updatePrimaryVertex` — default `True`. **Measured: about 0.1 ms.**

Normally the cache only stores at *secondary* hits, so an update path whose first bounce flies off
into the sky stores nothing at all — **measured** FNV desert: two thirds of the update budget,
wasted. With this on, the path also stores at the camera-visible surface it started from, valued as
the direct pass's lighting plus the sampled continuation. Every sky-bound path then contributes
something, and every camera-visible eligible surface gets fed by every update tile that lands on it.

**Leave it on unless you are chasing frame time.** **Measured** (user report): it improves quality,
and it costs **about 0.1 ms** — which is the same order as the whole cache's measured net benefit,
so this one option is roughly the price of the feature. That is why the **Performance** preset is
the only one that turns it off. **Expected, unmeasured**: the no-cell and below-floor shares both
fall, hit rate climbs toward the high 90s, `Path ends: sky` does not move (it is a query statistic).

It helps for two separate reasons, and only one of them is about the sky: sky-bound update paths
stop being wasted (outdoors only), *and* every camera-visible eligible surface gets a dense sample
every frame instead of waiting for a bounce ray to find it (everywhere, including interiors). So
turning it off outdoors thins the cache to near nothing — at that point
`rtx.integrateIndirectMode = 0` is the honest setting rather than a thin cache you still pay for.

> **Depends on RTXDI.** What it stores is the direct lighting the direct pass already computed for
> that pixel — and that pass uses RTXDI when `rtx.useRTXDI` is on and falls back to RIS sampling
> when it is off. So with RTXDI off, or configured differently, the value deposited into the cell
> changes: a different (noisier) direct-light estimate. With `rtx.enableDirectLighting` off it
> deposits essentially nothing but the continuation. It also **excludes** PSR pixels (seen through
> a mirror or glass), translucent primaries and view models, so a scene dominated by those gets
> less out of it than one that is not. And it takes a propagation slot (see `updateBounces`).
> Changing it clears the cache.
>
> **Quality risk, looked for and not seen:** every primary-deposited sample looks toward the
> camera, so on a glossy-but-eligible surface a cell can end up averaging the camera's particular
> view of its specular highlight — and unlike a secondary sample, that error does not average out,
> because every primary sample of a cell uses the same direction. It concentrates on **smooth
> metals**, whose outgoing radiance is entirely specular. Watch for a tint or brightness on cached
> glossy surfaces that follows the camera; debug view **583 (Cached Radiance)** shows it directly.
> The user has looked in Portal RTX and seen none. If you do see it, **raise `minRoughness`** — it
> already gates the primary deposit through the same test every other surface uses, so the remedy
> exists and needs no new option. `docs/SHARC-adaptive-2026-09-16.md` §7 works through why a
> primary-only roughness floor was considered and declined.

#### `rtx.sharc.updateSkyRetries` — default `1`, range 0–4. **Quality confirmed; cost unmeasured.**

When an update path's first bounce exits to the sky, re-aim it from a cosine lobe about the surface
normal and trace again, up to N times, so the budget lands on geometry more often outdoors.
**Expected** recovery of the wasted paths: 34% at 1 retry, 57% at 2, 71% at 3, 81% at 4.

No correctness concern — what a cell stores does not depend on how the ray reached it, so
re-aiming changes *which* cells get fed, not what they hold.

**Cost:** one extra ray on each path that missed. **Expected** ~0.66 extra segments per update path
at N=1, ~1.2 at N=2, on a pass that traces a few percent of the query's segments.

> **Depends on:** nothing gates it, but it only ever does anything where paths miss on the first
> bounce — i.e. under open sky. Pointless indoors. If `updatePrimaryVertex` is also on, the primary's
> sample completes before any retry, so neither is counted twice. Changing it clears the cache.

### Diagnostics

#### `rtx.sharc.measureGpuTime` — default `False`

Turns on the per-pass GPU timers and unlocks the rest of the panel.

> **Required by `collectQueryStats`** — with this off, no statistics are gathered at all, no matter
> what `collectQueryStats` says. Costs real time through lost GPU overlap. Off for final timing.

#### `rtx.sharc.collectQueryStats` — default `False`

Collects the eligibility, hit, reject and path-end counters, from a rotating 1-in-64 pixel sample
summed over 120 frames.

> **Depends on `measureGpuTime` being on.** **Measured** cost: about 1 ms. Off for final timing.

#### `rtx.sharc.logFallbackStats` — default `False`

Periodically writes to the log how often SHARC was selected but did not run, split by reason. Free
when off. Useful to confirm over a whole session rather than a moment that nothing is gating it.

### Backends (for A/B only — none of these change image quality)

`queryTraceRay` (default on), `queryRayGeneration`, `updateRayGeneration`, `deferredUpdates` select
how the passes are dispatched. They exist so implementations can be compared against each other with
identical cache policy.

> **Dependency worth knowing:** `queryRayGeneration` is **inert while `queryTraceRay` is on**, and
> its checkbox is hidden for that reason. Shader Execution Reordering also only applies with the
> TraceRay query backend. The panel prints which backend actually ran
> (`Query backend: TraceRay + SER` etc.) — trust that line over your config. All of them clear the
> cache on change.

### Things outside `rtx.sharc` that change SHARC

| Setting | What it does to SHARC |
|---|---|
| `rtx.integrateIndirectMode` | Must be `3`. Anything else and SHARC is entirely inactive. |
| device features (Int64, Int64 atomics, FP16, 16-bit storage, RayQuery) | Missing any → **silently traces ordinary paths**. Panel says "Unsupported device features". |
| `rtx.raytracedRenderTarget.enable` | When the game is actually drawing a ray-traced in-world screen, SHARC falls back for that **entire frame**. Shows in the fallback line. |
| `rtx.pathMaxBounces` (default 4) | Governs **query** paths only. Update paths use `rtx.sharc.updateBounces` instead. |
| `rtx.enableRussianRoulette` | Governs query paths only; update paths always run roulette-free. |
| `rtx.useRTXDI`, `rtx.enableDirectLighting` | Change what `updatePrimaryVertex` deposits (see above). |
| `rtx.di.enableSampleStealing` | The one site where SHARC and plain path tracing differ on a cache miss. **DLSS Ray Reconstruction's preset sets it `False` underneath you**, which is why the two are identical on a miss under RR. |
| `rtx.wboitEnabled`, opacity micromaps | SHARC has dedicated variants for both; no override needed. Changing either clears the cache. |
| Graphics presets | Fixed in this tree to leave an explicit SHARC selection alone. On an **older build**, clicking a preset silently moved you to NRC. |

---

## 6. The debug views

`rtx.debugView.debugViewIdx` 580–587, listed under "SHARC" in the Debug View panel. They are
written by the query stages only, so `rtx.integrateIndirectMode` must be `3`.

Most of them show **one** indirect vertex, selected by `Debug Knob [0]`: knob 0 is the first
indirect hit, knob 1 the second, and so on.

| Index | View | What it is for |
|---|---|---|
| **581** | **Rejection Reason** | **Start here when the cache seems dead.** The first failing eligibility term, colour-coded: red non-opaque, green medium, blue opacity < 1, yellow subsurface, magenta emissive, **cyan non-diffuse arrival (`allowSpecularPaths` off)**, white roughness below `minRoughness`, black eligible. A screen flooded cyan means the specular switch is off. A screen flooded magenta means `maxEmissiveLuminance` is too low. |
| **580** | **Outcome** | **The other diagnostic that matters.** What the cache actually did: green hit, red miss (eligible and far enough, but no usable cell), blue too close, yellow footprint too narrow, grey rejected (go to 581), black no vertex at this bounce. Green everywhere = working. Grey everywhere = eligibility problem. Red everywhere = population problem. |
| 582 | Too Close Guard | The distance guard at eligible vertices: green passed, red genuinely too close, yellow too close by the last re-trace leg only (grate, cutout, clipped geometry), magenta/cyan the red/yellow cases through a portal. Only worth opening if the panel's "Too close" number is large. |
| 583 | Cached Radiance | The radiance actually read from the cache where it ended a path (HDR — use EV100 or raise the max value). Black where the path never terminated on the cache. **This is how you judge blur and splotchiness**, not the panel — compare it at two `accumulationFrames` values in your dimmest interior. |
| 584 | Termination Bounce | Which bounce the cache ended the path at; 1 = first indirect hit, 0 = never. Set max value to `rtx.pathMaxBounces` or use a pseudo-colour mode. Shows where in the path the saving is coming from. |
| 585 | Grid: Cells | Hash-coloured cell at the vertex. Shows cell size and layout directly — the quickest way to see what `gridScale` is doing. |
| 586 | Grid: Level / Voxel Size / Last Leg | Raw values: R grid level, G cell size in world units, B last resolve leg. Pick a channel with a pseudo-colour mode and set the max value. |
| 587 | Grid: Cell Age | R frames accumulated, G frames since last sample (evicted at `staleFrames`), B sample count. Black = no cell yet. **This is where you see the sample floor problem directly** — if B sits at 2 and never climbs, those cells will never be readable. |

*Note: view 581's in-UI legend still describes the emissive case as "any non-zero emissive
radiance". That has not been true since `maxEmissiveLuminance` became a threshold; the view itself
is correct, only the legend text is stale.*

---

## 7. Is SHARC worth it?

**What it buys**

- **Lower noise** in indirect lighting, from averaging many samples into each cell.
- **Temporal stability** — the single clearest reported benefit. Indirect lighting stops crawling
  and boiling in motion.
- **Cheap multi-bounce depth.** A cache hit replaces the whole remaining tail of a path with one
  read, so extra bounces cost much less than they otherwise would.

**What it costs**

- **Spatial blur**, because one cell is one average over a region — features smaller than a cell
  get smoothed together.
- **Temporal blur and lag**, because a cell is also an average over `accumulationFrames`.
- Some GPU time for the update and resolve passes, which you can see on the panel.

**What it has actually bought here**

**Measured**, Portal RTX: about **0.1 ms**, with 78.5% of paths terminating on the cache. That
sounds contradictory until you notice why — the indirect pass was not the frame's bottleneck, so
ending paths early saved little. That is the honest figure, and it is the figure to expect: SHARC's
product in this project has been image stability, not speed.

**Measured**, FNV open desert: essentially nothing, about 0.1 ms for a cache that cannot help,
because two thirds of paths leave for the sky and the rest have a 1.01-segment tail with nothing
in it to cut.

**Never measured**: FNV interiors — which is 80%+ of that game and exactly the regime where SHARC
should pay. Anyone with the panel open in a shack, a corridor and a casino floor would be
generating the most useful data this feature has ever had.

**The case worth trying that nobody has tried.** Raise `rtx.pathMaxBounces` (default 4) with SHARC
on. Cheap depth is the thing SHARC provides that the alternatives do not, and nothing in this
project has tested it. The shape of the argument: a cache hit ends the path, so the extra bounce
budget is only actually spent where the cache missed — which should make higher bounce counts far
cheaper with SHARC than without. Note that this affects query paths only; `updateBounces` is
separate and unaffected. **Entirely unmeasured** — it is a hypothesis, and a good one.

**Versus the alternatives**, briefly and honestly. ReSTIR GI reuses samples across space and time
and applies clamps and boiling filters — less noise, but with correlation, lag and energy loss
baked in, and NVIDIA's own DLSS-RR preset is a list of measures to undo the correlation. NRC
replaces the path with a small neural network's prediction — almost no noise, but an approximation
error that a denoiser cannot see or fix. SHARC with no hits is plain, unbiased path tracing, and
under Ray Reconstruction the unbiased estimator's one weakness — variance — is the exact thing
being fixed downstream. That is the reported preference in this project, and it is consistent with
everything in the source, but it has not been settled by a controlled comparison.

---

## 8. What is measured and what is not

**Measured, Portal RTX** (enclosed test chambers, 2.54M sampled paths over 120 frames, cache age
1012), after the three config/code gates were cleared:

| | before | after |
|---|---|---|
| eligible surfaces | 0.2% | 73.8% |
| cache terminates | 0.2% | 78.5% |
| lookup hit rate | 58.6% | 99.2% |
| roulette path ends | 89.6% | 21.0% |
| segments/path | 1.71 | 1.12 |

Remaining rejects: roughness 15.8%, medium 5.8%, non-opaque 3.9%, emissive 0.8%. Too close 4.1% of
eligible (88.5% of that at the first bounce). Frame time gain: **0.1 ms**. Also measured: ray
portals active for a full 21-minute session with zero fallbacks.

**Measured, FNV open desert**: `Path ends: sky` 65.9%, 1.01 segments/path, cache terminates 23.3%,
hit rate 88.2% — at `accumulationFrames` 32, `minSampleCount` 2, `capacityLog2` 20,
`updateTileSize` 8, `updateBounces` 4, `gridScale` 50. Earlier readings over open scenery were as
low as ~20%.

**Measured (user reports, not captures)**: `minSampleCount = 2` removed the camera-movement glow;
`footprintGate` was a large improvement in cache coverage; enabling `allowSpecularPaths` helped
performance.

**Not measured — do not present any of these as results:**

- FNV interiors. Never sampled, at any setting. The recommended profile for them is reasoning, not
  measurement.
- `updateSkyRetries`' cost, in any scene. Its quality benefit is a user report, not a capture.
- Which scene `updatePrimaryVertex`'s 0.1 ms was read in, and which scene its quality gain was
  confirmed in. Both are user reports without a title attached, and several conclusions turn on
  the answer — see `docs/SHARC-adaptive-2026-09-16.md` §6.
- Any `gridScale` value other than 50.
- `capacityLog2` 22 versus 20.
- Frame-time effect of `footprintGate` or `minSampleCount`.
- `rtx.pathMaxBounces` raised with SHARC on.
- Cache insertion failures from a full hash bucket. Nothing counts them; the only test is to change
  `capacityLog2` and see whether "no cell" moves.

---

## Further reading

The reasoning behind all of the above, with `file:line` citations:

- `docs/SHARC-implementation-status.md` — the running log, including the Portal RTX and FNV results.
- `docs/SHARC-open-world-diagnosis-2026-09-16.md` — why the hit rate collapses outdoors, with the
  readability arithmetic.
- `docs/SHARC-quality-and-interior-profile-2026-09-16.md` — the interior profile, and exactly how
  SHARC-with-no-hits differs from plain path tracing.
- `docs/SHARC-grid-scale-2026-09-16.md` — what `gridScale` is, derived.
- `docs/SHARC-sky-budget-2026-09-16.md` — the two sky-recovery options and why they are unbiased.
- `docs/SHARC-adaptive-2026-09-16.md` — why SHARC does not tune itself to the scene, which options
  could change without clearing the cache (only three), and what `updatePrimaryVertex`'s 0.1 ms is
  made of.
- `docs/SHARC-path-correctness-audit-2026-09-15.md` — eligibility, emissive handling, the roughness
  split, the distance guard.
- `docs/SHARC-portals-investigation-2026-09-15.md` — ray portals (partly superseded by the
  portal-space cache key).
