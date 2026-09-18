#!/usr/bin/env python3
"""Compile and validate small Vulkan SHARC shader contracts.

This intentionally validates shader ABI and feature requirements only.  It does
not attempt to execute a cache or emulate a renderer.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout + result.stderr


def wrapper(pass_name: str, lock_fallback: bool, separate_emissive: bool) -> str:
    defines = ["SHARC_ENABLE_GLSL=0", "SHARC_UPDATE=0", "SHARC_QUERY=0"]
    if pass_name == "update":
        defines[1] = "SHARC_UPDATE=1"
    elif pass_name == "query":
        defines[2] = "SHARC_QUERY=1"
    if lock_fallback:
        defines.append("HASH_GRID_ENABLE_64_BIT_ATOMICS=0")
    else:
        defines.append("HASH_GRID_ENABLE_64_BIT_ATOMICS=1")
    if separate_emissive:
        defines.append("SHARC_SEPARATE_EMISSIVE=1")

    prefix = "\n".join(f"#define {item.replace('=', ' ', 1)}" for item in defines)
    # Use explicit set-3 bindings in every wrapper.  The SDK's buffer macros
    # expand these declarations without imposing a descriptor-set convention.
    resources = """
[[vk::binding(0, 3)]] RWStructuredBuffer<uint64_t> g_hash : register(u0, space3);
[[vk::binding(1, 3)]] RWStructuredBuffer<uint> g_lock : register(u1, space3);
[[vk::binding(2, 3)]] RWStructuredBuffer<SharcAccumulationData> g_accum : register(u2, space3);
[[vk::binding(3, 3)]] RWStructuredBuffer<SharcPackedData> g_resolved : register(u3, space3);
"""
    params = """
SharcParameters makeParameters()
{
    SharcParameters p;
    p.hashGridParameters.cameraPosition = float3(0, 0, 0);
    p.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    p.hashGridParameters.sceneScale = 50.0;
    p.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;
    p.hashGridData.capacity = 1024;
    p.hashGridData.hashEntriesBuffer = g_hash;
#if !HASH_GRID_ENABLE_64_BIT_ATOMICS && !HASH_GRID_COMPACT
    p.hashGridData.lockBuffer = g_lock;
#endif
    p.accumulationBuffer = g_accum;
    p.resolvedBuffer = g_resolved;
    p.radianceScale = 1000.0;
    return p;
}
"""
    if pass_name == "resolve":
        body = """
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    SharcParameters p = makeParameters();
    SharcResolveParameters r;
    r.cameraPositionPrev = float3(0, 0, 0);
    r.accumulationFrameNum = 32;
    r.responsiveFrameNum = 8;
    r.staleFrameNumMax = 64;
    r.frameIndex = 1;
    SharcResolveEntry(id.x, p, r);
}
"""
    elif pass_name == "update":
        body = """
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    SharcParameters p = makeParameters();
    SharcState s;
    SharcInit(s);
    SharcHitData h;
    h.positionWorld = float3(1, 1, 1);
    h.normalWorld = float3(0, 1, 0);
#if SHARC_SEPARATE_EMISSIVE
    h.emissive = float3(0, 0, 0);
#endif
    bool keepGoing = SharcUpdateHit(p, s, h, float3(0, 0, 0), 0.5);
    SharcSetThroughput(s, float3(1, 1, 1));
    if (!keepGoing) SharcUpdateMiss(p, s, float3(0, 0, 0));
}
"""
    else:
        body = """
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    SharcParameters p = makeParameters();
    SharcHitData h;
    h.positionWorld = float3(1, 1, 1);
    h.normalWorld = float3(0, 1, 0);
#if SHARC_SEPARATE_EMISSIVE
    h.emissive = float3(0, 0, 0);
#endif
    float3 radiance;
    if (SharcGetCachedRadiance(p, h, radiance, false))
        g_resolved[0].sampleData = asuint(radiance.x);
}
"""
    return f"{prefix}\n#include \"SharcCommon.h\"\n{resources}\n{params}\n{body}"


def require_disassembly(disassembly: str, lock_fallback: bool, pass_name: str) -> None:
    # Scalar layout must retain the SDK's structured element sizes.  The
    # offsets are observable in OpTypeStruct member decorations.
    if "ArrayStride 8" not in disassembly:
        raise RuntimeError(f"{pass_name}: missing 8-byte hash-entry stride")
    expected_accum = 16
    expected_resolved = 16
    if "SHARC_ENABLE_SH_ENCODING" in disassembly:
        expected_accum = 32
        expected_resolved = 24
    if f"ArrayStride {expected_accum}" not in disassembly:
        raise RuntimeError(f"{pass_name}: missing {expected_accum}-byte accumulation stride")
    if f"ArrayStride {expected_resolved}" not in disassembly:
        raise RuntimeError(f"{pass_name}: missing {expected_resolved}-byte resolved stride")
    if lock_fallback and pass_name.endswith("update") and "ArrayStride 4" not in disassembly:
        raise RuntimeError(f"{pass_name}: lock fallback has no 4-byte lock stride")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slangc", required=True, type=Path)
    parser.add_argument("--spirv-val", required=True, type=Path)
    parser.add_argument("--spirv-dis", required=True, type=Path)
    parser.add_argument("--include", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    include = args.include.resolve()
    args.output = args.output.resolve()
    if not (include / "SharcCommon.h").is_file():
        parser.error(f"--include lacks SharcCommon.h: {include}")
    args.output.mkdir(parents=True, exist_ok=True)

    cases = [("atomic64", False, False), ("lock", True, False), ("separate-emissive", False, True)]
    with tempfile.TemporaryDirectory(prefix="sharc-contract-") as temp:
        temp_path = Path(temp)
        for label, lock_fallback, separate_emissive in cases:
            for pass_name in ("update", "resolve", "query"):
                stem = f"{label}-{pass_name}"
                source = temp_path / f"{stem}.slang"
                output = args.output / f"{stem}.spv"
                disassembly_file = args.output / f"{stem}.spvasm"
                source.write_text(wrapper(pass_name, lock_fallback, separate_emissive), encoding="utf-8")
                command = [
                    str(args.slangc), "-target", "spirv", "-profile", "spirv_1_5",
                    "-entry", "main", "-stage", "compute", "-fvk-use-scalar-layout",
                    "-emit-spirv-directly", "-I", str(include), "-o", str(output), str(source),
                ]
                if not lock_fallback and pass_name == "update":
                    command += ["-capability", "spvInt64Atomics"]
                run(command, temp_path)
                run([str(args.spirv_val), "--target-env", "vulkan1.2", "--scalar-block-layout", str(output)], temp_path)
                disassembly = run([str(args.spirv_dis), str(output)], temp_path)
                disassembly_file.write_text(disassembly, encoding="utf-8")
                require_disassembly(disassembly, lock_fallback, stem)
                if not lock_fallback and pass_name == "update" and not re.search(r"Int64Atomics|AtomicCompareExchange", disassembly):
                    raise RuntimeError(f"{stem}: expected 64-bit atomic capability/use")
                print(f"PASS {stem}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
