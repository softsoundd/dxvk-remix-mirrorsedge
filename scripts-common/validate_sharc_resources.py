"""Check that compiled SHARC stages do not require other integrators' resources."""
import argparse
import re
import struct
from pathlib import Path


def bindings(blob):
    assert len(blob) % 4 == 0, 'Misaligned SPIR-V'
    words = struct.unpack('<' + 'I' * (len(blob) // 4), blob)
    assert words[0] == 0x07230203, 'Invalid SPIR-V magic'
    result = set()
    index = 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 0xffff
        assert count and index + count <= len(words), 'Invalid SPIR-V instruction'
        if opcode == 71 and count == 4 and words[index + 2] == 33:
            result.add(words[index + 3])
        index += count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dll', type=Path)
    args = parser.parse_args()
    header = (args.root / 'src/dxvk/shaders/rtx/pass/integrate/integrate_indirect_binding_indices.h').read_text()
    foreign = {int(value) for name, value in re.findall(r'#define\s+(INTEGRATE_INDIRECT_BINDING_\w+)\s+(\d+)', header)
               if '_NRC_' in name or '_RESTIR_GI_' in name}
    assert foreign, 'Missing resource definitions'
    source = (args.root / 'src/dxvk/rtx_render/rtx_pathtracer_integrate_indirect.cpp').read_text()
    names = set(re.findall(r'#include <rtx_shaders/(integrate_indirect_sharc_\w+)\.h>', source))
    files = [args.root / '_Comp64Release/src/dxvk/rtx_shaders' / (name + '.spv') for name in sorted(names)]
    assert files, 'No SHARC shader includes found'
    assert any('_wboit' in p.stem for p in files)
    dll = args.dll.read_bytes() if args.dll else None
    for path in files:
        blob = path.read_bytes()
        assert not bindings(blob) & foreign, f'{path.name}: requires removed resources'
        if '_fused' in path.stem:
            assert {190, 249, 250} <= bindings(blob), f'{path.name}: missing raw/primary outputs'
        else:
            assert not bindings(blob) & {249, 250}, f'{path.name}: unexpectedly requires fused outputs'

        if dll is not None:
            assert blob in dll, f'{path.name}: not embedded in final DLL'
    plain = (args.root / '_Comp64Release/src/dxvk/rtx_shaders/integrate_nee_plain.spv').read_bytes()
    plain_bindings = bindings(plain)
    assert not plain_bindings & {71, 82, 83}, 'Plain assembly requires removed backend descriptors'
    assert {80, 81} <= plain_bindings, 'Plain assembly missing primary lighting outputs'
    if dll is not None:
        assert plain in dll, 'Plain assembly not embedded in final DLL'
    print('PASS plain assembly: foreign descriptors absent, primary lighting outputs retained')
    print(f'PASS {len(files)} compiled SHARC stages: {len(foreign)} foreign descriptors absent'
          + (', final DLL embedding verified' if dll is not None else ''))


if __name__ == '__main__':
    main()
