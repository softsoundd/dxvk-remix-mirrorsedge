# Local config for scripts/regen-docs.ps1.
#
# Copy this file to `regen-docs.local.ps1` (which is gitignored) and
# point `$GameDir` at a directory where a Remix-API-consuming game is
# deployed (i.e. where `d3d9.dll` from this fork has been installed and
# the game launches without crashing on first frame).
#
# The driver launches `$GameDir\$GameExe` with
# DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD=1 set, waits for the runtime
# to write `RtxOptions.md` into `$GameDir`, then terminates the game.

$GameDir = 'C:\path\to\your\game\directory'
$GameExe = 'YourGameExecutable.exe'
