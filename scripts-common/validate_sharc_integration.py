#!/usr/bin/env python3
"""Validate the compiled SHARC integration SPIR-V contracts.

This is deliberately a static ABI check.  It does not execute shaders or
modify the build tree; ``--shader-dir`` is expected to contain the generated
SPIR-V files from a completed build.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def disassemble(path: Path, tool: Path) -> str:
    p = subprocess.run([str(tool), str(path)], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"spirv-dis failed for {path.name}: {p.stderr}")
    return p.stdout


def validate_spirv(path: Path, validator: Path) -> None:
    p = subprocess.run(
        [str(validator), "--target-env", "vulkan1.2", "--scalar-block-layout", str(path)],
        text=True, capture_output=True,
    )
    if p.returncode:
        raise RuntimeError(f"spirv-val failed for {path.name}: {p.stdout}\n{p.stderr}")


def binding_variables(text: str) -> dict[int, str]:
    return {int(binding): ident for ident, binding in re.findall(
        r"OpDecorate\s+(%\S+)\s+Binding\s+(\d+)", text)}


def strides(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"ArrayStride\s+(\d+)", text)}


def uses_wboit_compensation(text: str) -> bool:
    # An actual uniform access distinguishes the resolver branches after dead-code elimination.
    member = re.search(r'OpMemberName\s+(%\S+)\s+(\d+)\s+"wboitEnergyLossCompensation"', text)
    if not member:
        raise RuntimeError("missing WBOIT member in RaytraceArgs")
    pointers = re.findall(r'(%\S+)\s+=\s+OpTypePointer\s+Uniform\s+' + re.escape(member[1]) + r'\b', text)
    variables = set()
    for pointer in pointers:
        variables.update(re.findall(r'(%\S+)\s+=\s+OpVariable\s+' + re.escape(pointer) + r'\s+Uniform\b', text))
    indices = set(re.findall(r'(%\S+)\s+=\s+OpConstant\s+%\S+\s+' + member[2] + r'\b', text))
    return any(base in variables and index in indices for base, index in re.findall(
        r'OpAccessChain\s+%\S+\s+(%\S+)\s+(%\S+)', text))


def check(name: str, text: str) -> None:
    bv = binding_variables(text)
    b = set(bv)
    s = strides(text)
    if name.startswith("update"):
        required = {230, 231}
        if not required <= b:
            raise RuntimeError(f"{name}: missing SHARC bindings {sorted(required - b)}")
        if 232 in b:
            raise RuntimeError(f"{name}: query-only resolved binding 232 is present")
        if 8 not in s or 16 not in s:
            raise RuntimeError(f"{name}: expected hash/accumulation strides 8/16, got {sorted(s)}")
        if "Int64Atomics" not in text or "OpAtomic" not in text:
            raise RuntimeError(f"{name}: expected Int64Atomics capability and atomic operation")
        # The property that matters is that an update stage writes no render output. Banning the
        # OpImageWrite opcode outright was a proxy for that, and it is the wrong one: the deposit
        # and budget debug views are produced by the update pass because no query stage can see
        # the values, so they write the debug image on purpose. Resolve each write back to the
        # variable it loaded and name what is allowed instead, which also states the invariant
        # more precisely than the opcode ban did.
        loads = dict(re.findall(r"(%\w+)\s*=\s*OpLoad\s+%\w+\s+(%\w+)", text))
        for image in re.findall(r"OpImageWrite\s+(%\w+)", text):
            target = loads.get(image, image)
            if target != "%DebugView":
                raise RuntimeError(f"{name}: writes image {target}, only %DebugView is permitted")
        if 190 in b:
            raise RuntimeError(f"{name}: must not bind the final indirect image")
    elif name.startswith("query"):
        required = {230, 232}
        if not required <= b:
            raise RuntimeError(f"{name}: missing SHARC bindings {sorted(required - b)}")
        if "Int64Atomics" in text:
            raise RuntimeError(f"{name}: query must not perform atomic64 cache updates")
        if 190 not in b or "OpImageWrite" not in text:
            raise RuntimeError(f"{name}: query must write the final indirect image")
    elif name == "resolve":
        required = {230, 231, 232}
        if not required <= b:
            raise RuntimeError(f"{name}: missing SHARC bindings {sorted(required - b)}")
        if not {8, 16} <= s:
            raise RuntimeError(f"{name}: expected SHARC strides 8/16, got {sorted(s)}")
    elif name.startswith("baseline"):
        if b & {230, 231, 232}:
            raise RuntimeError(f"{name}: baseline unexpectedly declares SHARC binding")
        if "Int64Atomics" in text:
            raise RuntimeError(f"{name}: baseline unexpectedly requires Int64Atomics")

    if "raygen" in name:
        if not re.search(r"OpEntryPoint RayGeneration", text):
            raise RuntimeError(f"{name}: missing ray-generation entry point")
        if "OpTraceRayKHR" in text or "OpRayQueryInitializeKHR" not in text:
            raise RuntimeError(f"{name}: must use inline RayQuery without hit shaders")

    if name.startswith("query"):
        if (235 in b) != ("stats" in name.split("_")):
            raise RuntimeError(f"{name}: incorrect statistics binding")
        if "raygen" not in name and not re.search(r"OpExecutionMode\s+%\S+\s+LocalSize\s+8\s+8\s+1", text):
            raise RuntimeError(f"{name}: expected 8x8 compute workgroup")

    if name.startswith(("update", "query", "baseline")):
        if uses_wboit_compensation(text) != name.endswith("wboit"):
            raise RuntimeError(f"{name}: incorrect compiled particle resolver")

    # Updates read thread-task image 82 for the incoming PDF. Its RW image
    # declaration need not be NonWritable; absence of OpImageWrite proves isolation.
    # Query retains the ordinary indirect image and NEE feedback outputs.
    forbidden = {83, 170, 171, 172, 190}
    if name.startswith("update") and b & forbidden:
        raise RuntimeError(f"{name}: forbidden output binding(s) present: {sorted(b & forbidden)}")
    if name.startswith("update"):
        for slot in (80, 81):
            ident = bv.get(slot)
            if ident and not re.search(rf"OpDecorate\s+{re.escape(ident)}\s+NonWritable", text):
                raise RuntimeError(f"{name}: NEE cache binding {slot} is writable")


def payload_signature(text: str) -> tuple:
    definitions = dict(re.findall(r"^\s*(%\S+)\s+=\s+(Op(?:Type\w+|Constant)\b[^\r\n]*)", text, re.MULTILINE))
    payloads = re.findall(r"OpTypePointer\s+(?:Incoming)?RayPayloadKHR\s+(%\S+)", text)
    if not payloads:
        raise RuntimeError("missing ray payload")

    def shape(ident: str) -> tuple:
        return tuple(shape(token) if token.startswith("%") else token
                     for token in definitions[ident].split())

    signatures = {shape(ident) for ident in payloads}
    if len(signatures) != 1:
        raise RuntimeError("inconsistent payload types within a shader")
    return signatures.pop()


def check_trace(name: str, text: str, stage: str) -> None:
    b = set(binding_variables(text))
    if not re.search(r"OpEntryPoint\s+" + stage + r"KHR\b", text):
        raise RuntimeError(f"{name}: wrong ray tracing stage")
    if 231 in b or "Int64Atomics" in text:
        raise RuntimeError(f"{name}: query contains cache accumulation writes")
    if (235 in b) != ("stats" in name.split("_")):
        raise RuntimeError(f"{name}: incorrect statistics binding")
    if stage == "RayGeneration":
        if 190 not in b or "OpImageWrite" not in text:
            raise RuntimeError(f"{name}: missing final indirect image output")
        if b & {230, 232}:
            raise RuntimeError(f"{name}: cache lookup should run in hit/miss stages")
        if "ser" in name.split("_"):
            required_ops = ("OpHitObjectTraceRayNV", "OpReorderThreadWithHitObjectNV", "OpHitObjectExecuteShaderNV")
            if not all(op in text for op in required_ops) or "OpTraceRayKHR" in text:
                raise RuntimeError(f"{name}: expected SER hit-object trace/reorder/execute")
        elif "OpTraceRayKHR" not in text or "OpHitObjectTraceRayNV" in text:
            raise RuntimeError(f"{name}: expected ordinary TraceRay")
    else:
        if not {230, 232} <= b or not {8, 16} <= strides(text):
            raise RuntimeError(f"{name}: missing cache lookup bindings or strides")
        if "OpTraceRayKHR" in text or "OpHitObjectTraceRayNV" in text:
            raise RuntimeError(f"{name}: recursive TraceRay requires a deeper pipeline")
        if "OpRayQueryInitializeKHR" not in text:
            raise RuntimeError(f"{name}: expected inline visibility queries")
        if uses_wboit_compensation(text) != name.endswith("wboit"):
            raise RuntimeError(f"{name}: incorrect compiled particle resolver")
    if b & {51, 170, 171, 172}:
        raise RuntimeError(f"{name}: reservoir policy differs from inline SHARC")


def uniform_member_used(text: str, name: str) -> bool:
    """True when a RaytraceArgs member is actually accessed after dead-code elimination."""
    member = re.search(r'OpMemberName\s+(%\S+)\s+(\d+)\s+"' + re.escape(name) + '"', text)
    if not member:
        return False
    pointers = re.findall(r'(%\S+)\s+=\s+OpTypePointer\s+Uniform\s+' + re.escape(member[1]) + r'\b', text)
    variables = set()
    for pointer in pointers:
        variables.update(re.findall(r'(%\S+)\s+=\s+OpVariable\s+' + re.escape(pointer) + r'\s+Uniform\b', text))
    indices = set(re.findall(r'(%\S+)\s+=\s+OpConstant\s+%\S+\s+' + member[2] + r'\b', text))
    return any(base in variables and index in indices for base, index in re.findall(
        r'OpAccessChain\s+%\S+\s+(%\S+)\s+(%\S+)', text))


def declared_sharc_variants(root: Path) -> set:
    declared = set()
    for source in ("integrate_indirect.slang", "integrate_indirect_closesthit.rchit.slang", "integrate_indirect_miss.rmiss.slang"):
        text = (root / "src/dxvk/shaders/rtx/pass/integrate" / source).read_text(encoding="utf-8")
        declared.update(re.findall(r"^//!variant\s+(integrate_indirect_sharc_\S+)\.\w+\s*$", text, re.MULTILINE))
    return declared


def check_reservoir_guard(root: Path, shader_dir: Path, dis: Path, baseline_dir, baseline_dll) -> None:
    """Pin the SHARC stealing guard: SHARC stages must not enter the RTXDI stealing branch.

    Without RAB_HAS_RTXDI_RESERVOIRS a steal always fails and skips the NEE cache / RIS fallback,
    so SHARC stages (which never bind the reservoir) must compile the branch out entirely.
    """
    source = (root / "src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh").read_text(encoding="utf-8")
    if not re.search(r"#if ENABLE_SHARC && !defined\(RAB_HAS_RTXDI_RESERVOIRS\)\s*\n\s*if \(false\)", source):
        raise RuntimeError("integrator_indirect.slangh: SHARC reservoir guard around the stealing branch is missing")
    context = (root / "src/dxvk/rtx_render/rtx_context.cpp").read_text(encoding="utf-8")
    for needle in ("enableIndirectAlphaBlendShadows() && accelManager.hasAlphaBlendInstances()",
                   "enableUnorderedResolveInIndirectRays() && accelManager.getUnorderedInstanceCount() > 0"):
        if needle not in context:
            raise RuntimeError(f"rtx_context.cpp: scene gate missing: {needle}")
    declared = sorted(declared_sharc_variants(root))
    if not declared:
        raise RuntimeError("no SHARC variants declared")
    for stem in declared:
        path = shader_dir / (stem + ".spv")
        if not path.is_file():
            raise RuntimeError(f"declared SHARC variant not compiled: {stem}")
        text = disassemble(path, dis)
        if set(binding_variables(text)) & {10, 51}:
            raise RuntimeError(f"{stem}: SHARC stage binds the RTXDI reservoir or previous lights")
        if uniform_member_used(text, "enableRtxdiSampleStealing"):
            raise RuntimeError(f"{stem}: SHARC stage still enters the RTXDI stealing branch")
        if baseline_dir is not None:
            reference = baseline_dir / (stem + ".spv")
            if not reference.is_file():
                raise RuntimeError(f"{stem}: missing from baseline directory")
            if reference.read_bytes() != path.read_bytes():
                raise RuntimeError(f"{stem}: differs from baseline")
    print(f"PASS reservoir guard: {len(declared)} declared SHARC stages omit bindings 10/51 and the stealing flag"
          + (", byte-identical to baseline" if baseline_dir is not None else ""))
    legacy = disassemble(shader_dir / "integrate_indirect_neeCache_material_opaque_translucent_closestHit.spv", dis)
    if 51 not in set(binding_variables(legacy)) or not uniform_member_used(legacy, "enableRtxdiSampleStealing"):
        raise RuntimeError("legacy TraceRay closest hit lost RTXDI sample stealing")
    print("PASS legacy TraceRay closest hit retains the reservoir binding and sample stealing")
    if baseline_dll is not None:
        old = baseline_dll.read_bytes()
        for stem in ("integrate_indirect_rayquery_neeCache", "integrate_indirect_rayquery_neeCache_wboit",
                     "integrate_indirect_neeCache_material_opaque_translucent_closestHit",
                     "integrate_indirect_neeCache_material_rayportal_closestHit"):
            if (shader_dir / (stem + ".spv")).read_bytes() not in old:
                raise RuntimeError(f"{stem}: legacy shader differs from the baseline DLL")
        print("PASS legacy stages byte-identical to the baseline DLL")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shader-dir", required=True, type=Path)
    ap.add_argument("--spirv-dis", required=True, type=Path)
    ap.add_argument("--spirv-val", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repository root for source-level checks")
    ap.add_argument("--baseline-dir", type=Path, help="directory of SHARC blobs that must be byte-identical")
    ap.add_argument("--baseline-dll", type=Path, help="older DLL that must still embed the legacy stages")
    args = ap.parse_args()
    files = {
        "update": args.shader_dir / "integrate_indirect_sharc_update.spv",
        "update_deferred": args.shader_dir / "integrate_indirect_sharc_update_deferred.spv",
        "update_deferred4": args.shader_dir / "integrate_indirect_sharc_update_deferred4.spv",
        "update_raygen": args.shader_dir / "integrate_indirect_sharc_update_raygen.spv",
        "update_deferred_raygen": args.shader_dir / "integrate_indirect_sharc_update_deferred_raygen.spv",
        "update_deferred4_raygen": args.shader_dir / "integrate_indirect_sharc_update_deferred4_raygen.spv",
        "query": args.shader_dir / "integrate_indirect_sharc_query.spv",
        "query_stats": args.shader_dir / "integrate_indirect_sharc_query_stats.spv",
        "query_raygen": args.shader_dir / "integrate_indirect_sharc_query_raygen.spv",
        "query_raygen_stats": args.shader_dir / "integrate_indirect_sharc_query_raygen_stats.spv",
        "resolve": args.shader_dir / "sharc_resolve.spv",
        "baseline": args.shader_dir / "integrate_indirect_rayquery_neeCache.spv",
    }
    for name, path in list(files.items()):
        if name.startswith(("update", "query")):
            files[name + "_wboit"] = path.with_stem(path.stem + "_wboit")
    files["baseline_wboit"] = args.shader_dir / "integrate_indirect_rayquery_neeCache_wboit.spv"
    for name, stem in (("baseline_closesthit", "integrate_indirect_neeCache_pom_material_rayportal_closestHit"),
                       ("baseline_miss", "integrate_indirect_miss_neeCache")):
        files[name] = args.shader_dir / (stem + ".spv")
        files[name + "_wboit"] = args.shader_dir / (stem + "_wboit.spv")
    compiled = {}
    for name, path in files.items():
        if not path.is_file():
            raise RuntimeError(f"missing {name} shader: {path}")
        validate_spirv(path, args.spirv_val)
        compiled[name] = disassemble(path, args.spirv_dis)
        check(name, compiled[name])
        print(f"PASS {name}: {path.name}")
    trace_payload = None
    for stage in ("trace", "closesthit", "miss", "closesthit_no_pom", "closesthit_no_portals",
                  "closesthit_no_portals_no_pom", "miss_no_portals"):
        for ser in ((False, True) if stage == "trace" else (False,)):
            for stats in (False, True):
                for wboit in (False, True):
                    suffix = ("_stats" if stats else "") + ("_wboit" if wboit else "")
                    name = stage + ("_ser" if ser else "") + suffix
                    path = args.shader_dir / ("integrate_indirect_sharc_query_" + name + ".spv")
                    validate_spirv(path, args.spirv_val)
                    text = disassemble(path, args.spirv_dis)
                    check_trace(name, text, {"trace": "RayGeneration", "closesthit": "ClosestHit", "miss": "Miss"}[stage.split("_")[0]])
                    signature = payload_signature(text)
                    if trace_payload is None:
                        trace_payload = signature
                    elif trace_payload != signature:
                        raise RuntimeError(f"{name}: ray payload ABI differs across shader stages/variants")
                    inline_bindings = set(binding_variables(compiled["query_raygen" + suffix]))
                    extra = set(binding_variables(text)) - inline_bindings
                    if extra:
                        raise RuntimeError(f"{name}: descriptors exceed inline SHARC layout: {sorted(extra)}")
                    print(f"PASS {name}: stage, payload ABI, descriptors, tracing and resolver")
    check_reservoir_guard(args.root, args.shader_dir, args.spirv_dis, args.baseline_dir, args.baseline_dll)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
