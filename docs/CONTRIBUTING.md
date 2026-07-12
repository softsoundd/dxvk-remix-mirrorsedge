# Contributing to Remix Plus

Thanks for being here! Remix Plus is a small, focused fork — every
contribution that lands moves it forward in a real way. This guide
walks you through setting up, working on something, and sending it
back as a pull request. It's meant to be readable end-to-end in a
few minutes.

If you get stuck, open an issue or ask on the
[RTX Remix Discord](https://discord.gg/c7J6gUhXMk). Friendly
questions are welcome.

## Why this fork exists

NVIDIA's stock `dxvk-remix` was tuned around a small set of
demonstration mods. The community **gmod-rtx** fork extended the
Remix SDK API for real game integrations — batched mesh and light
creation, plugin-injected game state, UI state plumbing, VRAM
control, and so on — but it lives in a Garry's Mod-flavored tree
that's hard to use as a base for other engines.

**Remix Plus** brings those SDK extensions back into a clean,
NVIDIA-rebase-friendly fork. The goal is a maintained codebase that
plugin developers and game integrations can build on without
inheriting gmod-specific assumptions, and that can absorb upstream
NVIDIA improvements without two days of merge pain every time.

If you write Remix plugins, integrate Remix into a game, or just
want to make this fork better — you're in the right place.

## Setting up your fork

1. **Fork** [`RemixProjGroup/dxvk-remix`](https://github.com/RemixProjGroup/dxvk-remix)
   on GitHub.
2. **Clone** your fork locally with submodules:
   ```
   git clone --recursive https://github.com/<your-username>/dxvk-remix.git
   ```
   If you forgot `--recursive`, run `git submodule update --init --recursive`
   inside the clone.
3. *(Optional)* Add the canonical repo as an extra remote so you can
   pull updates from it:
   ```
   git remote add canonical https://github.com/RemixProjGroup/dxvk-remix.git
   ```

The canonical repo's default branch is `main`. Your
fork has a tracking copy of that branch — it's where shipped work
lives, and it's the target of every PR.

## Building

Remix Plus builds on Windows with MSVC and ninja. The full build
takes 10–15 minutes cold; incremental builds are 1–3 minutes.

### Requirements

- Windows 10 or 11
- [Visual Studio 2019](https://visualstudio.microsoft.com/vs/older-downloads/)
  (2022 may work but isn't actively tested)
- [Windows SDK 10.0.19041.0](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/)
- [Meson 1.8.2+](https://mesonbuild.com/)
- [Vulkan SDK 1.4.313.2+](https://vulkan.lunarg.com/sdk/home#windows)
- [Python 3.9+](https://www.python.org/downloads/) (not the Microsoft
  Store version)
- [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35)
  (you may already have it from any D3D9 game)

Only x64 is supported.

### First-time setup

PowerShell needs to allow local scripts. Run once in an elevated
PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

### Build

From the project root, in PowerShell:

```powershell
.\scripts\build.ps1
```

This builds the release flavor into `_Comp64Release/` and installs
`d3d9.dll` to `_output/`. Use `-Flavor debug` or `-Flavor debugoptimized`
for the instrumented flavors, `-Clean` to wipe the build dir first, and
`-EnableTracy` to enable the Tracy profiler:

```powershell
.\scripts\build.ps1 -Flavor debugoptimized
.\scripts\build.ps1 -Clean -EnableTracy
```

The build copies `d3d9.dll` to any game targets configured in
`gametargets.conf` (copy `gametargets.example.conf` and edit paths
the first time).

### When dependencies change

If a dependency path changes (new Vulkan SDK, etc.), run
`meson --reconfigure` inside the build folder.

### Profiling

Tracy profiling is supported. Enable with
`meson --reconfigure -D enable_tracy=true`, rebuild, then connect
[Tracy v0.8.7](https://github.com/wolfpld/tracy/releases/download/v0.8/Tracy-0.8.7z)
to a running session.

## Making changes — general guidance

### Branch on your fork

Use whatever branch name makes sense to you — `fix-tonemap-clamp`,
`add-vram-stat`, `feature/lighting-tweak`, anything. The branch
name on your fork doesn't affect review.

### Keep PRs focused

One thing per PR. A PR that fixes a tonemap bug should not also
touch unrelated capture code. Reviewers can move quickly through
small, focused PRs; mixed PRs stall.

### Small commits with clear messages

Each commit should be a self-contained step. The first line is a
short summary in present tense ("add Lottes 2016 operator", not
"added" or "this PR adds"). The body, when needed, explains the
*why* — the *what* lives in the diff.

We use conventional-commit prefixes (`fork(...)`, `feat(...)`,
`fix(...)`, `docs(...)`, etc.) but they're a soft convention, not
a check. Match the style of recent commits with `git log --oneline`.

### No AI co-author trailers

Don't add `Co-Authored-By: Claude ...` or similar trailers unless
the maintainer explicitly asks for it. Authorship stays with the
human contributor.

### Credit yourself

When your PR adds something visible, add your name to
[`src/dxvk/imgui/dxvk_imgui_about.cpp`](../src/dxvk/imgui/dxvk_imgui_about.cpp)
under "GitHub Contributors", organized A–Z by last name. This
shows up in the in-game About panel.

## Fork-touchpoint discipline (the one rigid rule)

This is the rule that keeps Remix Plus rebaseable against NVIDIA's
upstream `dxvk-remix`. Read it before touching any file that came
from upstream.

### The pattern

When you need to add fork logic that lives inside an upstream-owned
file, **don't write it inline**. Instead:

1. Put the substantive code in a fork-owned file
   (`src/dxvk/rtx_render/rtx_fork_<subsystem>.cpp/h`).
2. Declare the entry point in the `fork_hooks::` namespace
   (`src/dxvk/rtx_render/rtx_fork_hooks.h`).
3. Leave a single one-line dispatch in the upstream file:

```cpp
// In an upstream file like rtx_scene_manager.cpp:
void SceneManager::submitDraw(const DrawCmd& cmd) {
  fork_hooks::onExternalDraw(*this, cmd);  // ← the touchpoint
  // ... rest of NVIDIA's original function, untouched ...
}
```

Why? Upstream files get touched on every NVIDIA release. Each line
of fork logic inside an upstream file is a future merge conflict.
A one-line hook becomes trivial to re-locate when upstream rewrites
the function around it. Inline blocks, by contrast, drown in
conflict markers.

### When inline is okay

If the change is genuinely tiny (under ~20 LOC) and structurally
can't be lifted out — adding a struct field, flipping an enum bit,
inserting a 2-line guard — inline is fine.

### Always update the fridge list

**Every commit that edits an upstream file must update
[`docs/fork-touchpoints.md`](fork-touchpoints.md) in the same
commit.** This is the authoritative inventory of every upstream
file the fork touches, with Hook-vs-Inline classification. The PR
template includes a checkbox reminder.

Run the audit script before committing if you want to double-check:

```
FORK_POINT=<sha> ./scripts/audit-fork-touchpoints.sh
```

Replace `<sha>` with the upstream NVIDIA fork-point.

## Code style

Two style guides apply:

- C++ — [`docs/CONTRIBUTING-style-guide.md`](CONTRIBUTING-style-guide.md)
- Shaders — [`src/dxvk/shaders/rtx/README.md`](../src/dxvk/shaders/rtx/README.md)

Headlines: 2-space indent, braces on the same line, `m_` for
member variables, `p` for pointer variables, `camelCase` for
functions and variables, `PascalCase` for types, `kConstant` for
constants, `UPPER_CASE` for macros.

When you change a non-RTX upstream file, wrap the diverging block
with comment fences:

```cpp
// NV-DXVK start: Brief description of change
// ... changed code ...
// NV-DXVK end
```

## Submitting your PR

1. **Build clean.** Compile-check release flavor; you should see
   exit code 0 and zero compile errors. The `rtx-build` skill (or
   the PowerShell command above) does this.
2. **Fast-forward** your fork's `main` to your
   feature-branch tip. Fast-forward only — never `--force` on
   `main`.
3. **Push** both your feature branch and your updated
   `main` to your fork.
4. **Open a PR** from your fork's `main` →
   canonical's `main`.
5. **Squash on merge.** Maintainers squash PRs at merge time, so
   each merged PR shows up as a single commit on the canonical
   branch. You don't need to squash locally.

## Bug reports

Bugs in **Remix Plus specifically** go in this repo's issue tracker.

Bugs in the broader RTX Remix runtime (not the dxvk-remix fork)
belong in [NVIDIAGameWorks/rtx-remix](https://github.com/NVIDIAGameWorks/rtx-remix/issues).
If you're not sure which one applies, file here and we'll redirect.

## Credits and upstream attribution

Remix Plus stands on the shoulders of:

- [DXVK](https://github.com/doitsujin/dxvk) — the original D3D9 → Vulkan
  translation layer this whole project descends from.
- [NVIDIA `dxvk-remix`](https://github.com/NVIDIAGameWorks/dxvk-remix) —
  the path-traced remastering fork of DXVK.
- The **gmod-rtx community fork** — where most of the SDK extensions
  carried by Remix Plus originated.

Thanks to everyone whose work makes this possible.
