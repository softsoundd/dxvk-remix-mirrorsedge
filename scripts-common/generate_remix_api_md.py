"""Generates RemixApiSurface.md from public/include/remix/remix_c.h.

Pure Python 3 stdlib. Hand-rolled tokenizer is acceptable because
remix_c.h is regular C with no preprocessor branching mid-declaration.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Declaration:
    kind: str          # 'define' | 'typedef' | 'enum' | 'struct' | 'forward'
    body: str
    leading_comment: str = ''


@dataclass
class Define:
    name: str
    value: str
    params: Optional[List[str]] = None  # None = object-like; list = function-like
    leading_comment: str = ''


def parse_define(decl: Declaration) -> Define:
    """Parse a Declaration with kind='define' into a Define object.

    Handles both object-like macros (#define FOO 42) and function-like
    macros (#define MAKE(a, b) ...). Joins backslash-continued lines.
    """
    # Join continuation lines.
    joined = re.sub(r'\\\s*\n', ' ', decl.body)
    # Strip the `#define` prefix.
    m = re.match(r'\s*#\s*define\s+(\w+)(\([^)]*\))?\s*(.*)', joined, flags=re.DOTALL)
    if not m:
        raise ValueError(f'Could not parse #define: {decl.body!r}')
    name = m.group(1)
    params_raw = m.group(2)
    value = m.group(3).strip()
    params = None
    if params_raw is not None:
        inner = params_raw[1:-1].strip()
        params = [p.strip() for p in inner.split(',')] if inner else []
    return Define(
        name=name,
        value=value,
        params=params,
        leading_comment=decl.leading_comment,
    )


@dataclass
class SimpleTypedef:
    name: str
    aliased_type: str
    leading_comment: str = ''


def parse_simple_typedef(decl: Declaration) -> SimpleTypedef:
    """Parse a Declaration with kind='typedef' or 'forward' into a SimpleTypedef.

    Handles scalar typedefs (typedef uint32_t remixapi_Bool;),
    handle typedefs (typedef struct X_T* X;), and forward struct declarations
    (typedef struct Foo Foo;).
    """
    body = re.sub(r'\s+', ' ', decl.body).strip()
    # Strip trailing semicolon.
    body = body.rstrip(';').strip()
    # `typedef <aliased ...> <name>` — name is the last word.
    m = re.match(r'typedef\s+(.+?)\s+(\w+)$', body)
    if not m:
        raise ValueError(f'Could not parse typedef: {decl.body!r}')
    aliased = m.group(1).strip()
    name = m.group(2).strip()
    return SimpleTypedef(
        name=name,
        aliased_type=aliased,
        leading_comment=decl.leading_comment,
    )


@dataclass
class FnPtrTypedef:
    name: str
    return_type: str
    params: List[str]
    leading_comment: str = ''


_FN_PTR_RE = re.compile(
    r'typedef\s+(?P<ret>[\w\s\*]+?)\s*'
    r'\(\s*REMIXAPI_PTR\s*\*\s*(?P<name>PFN_\w+)\s*\)\s*'
    r'\(\s*(?P<params>[^)]*)\)\s*;',
    flags=re.DOTALL,
)


@dataclass
class EnumMember:
    name: str
    value: str
    comment: str = ''


@dataclass
class Enum:
    name: str
    members: List[EnumMember] = field(default_factory=list)
    leading_comment: str = ''


@dataclass
class StructField:
    name: str
    type: str
    comment: str = ''


@dataclass
class Struct:
    name: str
    fields: List[StructField] = field(default_factory=list)
    is_info_struct: bool = False
    leading_comment: str = ''


def parse_fn_ptr_typedef(decl: Declaration) -> Optional[FnPtrTypedef]:
    """Parse a Declaration with kind='typedef' for a function-pointer typedef.

    Returns a FnPtrTypedef if the declaration is a function-pointer typedef of the
    form `typedef <return> (REMIXAPI_PTR* PFN_xxx)(<params>);`.
    Returns None if the declaration is not a function-pointer typedef.
    """
    body = re.sub(r'\s+', ' ', decl.body).strip()
    m = _FN_PTR_RE.search(body)
    if not m:
        return None
    return_type = m.group('ret').strip()
    name = m.group('name').strip()
    raw_params = m.group('params').strip()
    if raw_params == '':
        params: List[str] = []
    else:
        params = [p.strip() for p in raw_params.split(',')]
    return FnPtrTypedef(
        name=name,
        return_type=return_type,
        params=params,
        leading_comment=decl.leading_comment,
    )


def parse_enum(decl: Declaration) -> Enum:
    """Parse a Declaration with kind='enum' into an Enum object.

    Handles typedef enum { ... } name; blocks, extracting the enum name,
    members with their values (or '(implicit)' if absent), and member comments
    (captured from preceding // lines).
    """
    body = decl.body
    open_brace = body.index('{')
    close_brace = body.rindex('}')
    inner = body[open_brace + 1:close_brace]
    # Name follows the close brace.
    after = body[close_brace + 1:].strip().rstrip(';').strip()
    name = after.split()[0] if after else ''

    members: List[EnumMember] = []
    pending_comment = ''
    for raw_line in inner.split('\n'):
        line = raw_line.strip().rstrip(',').strip()
        if not line:
            pending_comment = ''
            continue
        if line.startswith('//'):
            pending_comment = (pending_comment + ' ' + line.lstrip('/').strip()).strip()
            continue
        if '=' in line:
            mem_name, mem_value = [p.strip() for p in line.split('=', 1)]
            value = mem_value
        else:
            mem_name = line
            value = '(implicit)'
        # Drop trailing inline comments on the value.
        value = re.sub(r'\s*//.*$', '', value).strip()
        members.append(EnumMember(
            name=mem_name,
            value=value,
            comment=pending_comment,
        ))
        pending_comment = ''

    return Enum(
        name=name,
        members=members,
        leading_comment=decl.leading_comment,
    )


def parse_struct(decl: Declaration) -> Struct:
    """Parse a Declaration with kind='struct' into a Struct object.

    Handles typedef struct { ... } name; blocks, extracting the struct name,
    fields with their types, and field comments (captured from preceding //
    lines or trailing inline comments).
    """
    body = decl.body
    open_brace = body.index('{')
    close_brace = body.rindex('}')
    inner = body[open_brace + 1:close_brace]
    # Name follows the close brace.
    after = body[close_brace + 1:].strip().rstrip(';').strip()
    name = after.split()[0] if after else ''

    fields: List[StructField] = []
    pending_comment = ''
    for raw_line in inner.split('\n'):
        line = raw_line.strip()
        if not line:
            pending_comment = ''
            continue
        if line.startswith('//'):
            pending_comment = (pending_comment + ' ' + line.lstrip('/').strip()).strip()
            continue
        # Drop trailing inline comments from the field line itself, capture
        # them as the field's comment.
        inline_comment = ''
        m_inline = re.search(r'//(.*)$', line)
        if m_inline:
            inline_comment = m_inline.group(1).strip()
            line = line[:m_inline.start()].rstrip()
        line = line.rstrip(';').strip()
        if not line:
            pending_comment = ''
            continue
        # Last whitespace-separated token is the field name; the rest is the
        # type. Handle arrays like `float matrix[3][4]`.
        m_field = re.match(r'(?P<type>.+?)\s+(?P<name>\w+(?:\[[^\]]+\])*)\s*$', line)
        if not m_field:
            # Skip lines that aren't field declarations (rare).
            pending_comment = ''
            continue
        field_type = m_field.group('type').strip()
        field_name = m_field.group('name').strip()
        comment_text = pending_comment or inline_comment
        fields.append(StructField(
            name=field_name,
            type=field_type,
            comment=comment_text,
        ))
        pending_comment = ''

    is_info = name.endswith('Info') or name.endswith('EXT')

    return Struct(
        name=name,
        fields=fields,
        is_info_struct=is_info,
        leading_comment=decl.leading_comment,
    )


def emit_defines_section(defines: List[Define]) -> str:
    out = ['## Macros and version', '']
    out.append('| Name | Value |')
    out.append('| :-- | :-- |')
    for d in defines:
        signature = d.name
        if d.params is not None:
            signature = f'{d.name}({", ".join(d.params)})'
        value = d.value.replace('|', '\\|') if d.value else ''
        out.append(f'| `{signature}` | `{value}` |')
    return '\n'.join(out)


def emit_simple_typedefs_section(typedefs: List[SimpleTypedef]) -> str:
    out = ['## Typedefs and handles', '']
    out.append('| Name | Aliased type |')
    out.append('| :-- | :-- |')
    for t in typedefs:
        out.append(f'| `{t.name}` | `{t.aliased_type}` |')
    return '\n'.join(out)


def emit_enum(e: Enum) -> str:
    out = [f'### `{e.name}`', '']
    if e.leading_comment:
        out.append(e.leading_comment)
        out.append('')
    out.append('| Member | Value | Notes |')
    out.append('| :-- | :-: | :-- |')
    for m in e.members:
        notes = m.comment.replace('|', '\\|')
        out.append(f'| `{m.name}` | `{m.value}` | {notes} |')
    return '\n'.join(out)


def emit_struct(s: Struct) -> str:
    out = [f'### `{s.name}`', '']
    if s.leading_comment:
        out.append(s.leading_comment)
        out.append('')
    out.append('| Type | Field | Notes |')
    out.append('| :-- | :-- | :-- |')
    for f in s.fields:
        notes = f.comment.replace('|', '\\|')
        out.append(f'| `{f.type}` | `{f.name}` | {notes} |')
    return '\n'.join(out)


def emit_fn_ptr(f: FnPtrTypedef) -> str:
    out = [f'### `{f.name}`', '']
    out.append(f'Returns: `{f.return_type}`')
    out.append('')
    out.append('Parameters:')
    out.append('')
    if not f.params or f.params == ['void']:
        out.append('- *(none)*')
    else:
        for p in f.params:
            out.append(f'- `{p}`')
    return '\n'.join(out)


def emit_full_document(decls: List[Declaration], header_path: str) -> str:
    defines: List[Define] = []
    simple_typedefs: List[SimpleTypedef] = []
    fn_ptrs: List[FnPtrTypedef] = []
    enums: List[Enum] = []
    structs: List[Struct] = []
    interface_struct: Optional[Struct] = None

    for d in decls:
        if d.kind == 'define':
            defines.append(parse_define(d))
        elif d.kind == 'enum':
            enums.append(parse_enum(d))
        elif d.kind == 'struct':
            s = parse_struct(d)
            if s.name == 'remixapi_Interface':
                interface_struct = s
            else:
                structs.append(s)
        elif d.kind == 'typedef':
            fn = parse_fn_ptr_typedef(d)
            if fn is not None:
                fn_ptrs.append(fn)
            else:
                try:
                    simple_typedefs.append(parse_simple_typedef(d))
                except ValueError:
                    pass  # non-PFN fn-ptr typedefs (e.g. __stdcall*) — skip
        elif d.kind == 'forward':
            try:
                simple_typedefs.append(parse_simple_typedef(d))
            except ValueError:
                pass

    parts = [
        '# Remix C API Surface',
        '',
        '> Auto-generated from [`public/include/remix/remix_c.h`](public/include/remix/remix_c.h)',
        '> by `scripts-common/generate_remix_api_md.py`. **Do not hand-edit.**',
        '> Regenerate via `scripts/regen-docs.ps1` from the repo root.',
        '',
        'This page is a flat exhaustive reference of every type, function pointer,',
        'enum, struct, and macro that a plugin or host application compiles',
        'against. For the conceptual / mental-model walkthrough of how to USE',
        'the API, see [`docs/RemixApi.md`](docs/RemixApi.md).',
        '',
        '---',
        '',
        emit_defines_section(defines),
        '',
        '---',
        '',
        emit_simple_typedefs_section(simple_typedefs),
        '',
        '---',
        '',
        '## Enums',
        '',
    ]
    for e in enums:
        parts.append(emit_enum(e))
        parts.append('')

    parts.extend(['---', '', '## Structs', ''])
    for s in structs:
        parts.append(emit_struct(s))
        parts.append('')

    parts.extend(['---', '', '## Function-pointer typedefs', ''])
    for fn in fn_ptrs:
        parts.append(emit_fn_ptr(fn))
        parts.append('')

    if interface_struct is not None:
        parts.extend(['---', '', '## `remixapi_Interface` table', ''])
        parts.append(
            'The function-pointer table that `remixapi_InitializeLibrary`'
            ' fills out. Each field points at the matching `PFN_*` typedef'
            ' documented in the previous section.')
        parts.append('')
        parts.append(emit_struct(interface_struct))
        parts.append('')

    return '\n'.join(parts)


def strip_block_comments(src: str) -> str:
    """Remove /* ... */ comments but keep // line comments intact."""
    return re.sub(r'/\*.*?\*/', '', src, flags=re.DOTALL)


def tokenize_top_level(src: str) -> List[Declaration]:
    """Split a C source string into top-level declarations.

    Recognizes: #define ..., typedef ... ;, typedef enum { ... } ...;,
    typedef struct { ... } ...;, and bare `typedef struct Foo Foo;` forward
    declarations. Captures any // comment lines immediately preceding the
    declaration into `leading_comment`.
    """
    src = strip_block_comments(src)
    lines = src.split('\n')

    decls: List[Declaration] = []
    pending_comment: List[str] = []
    i = 0
    n = len(lines)

    def join_comment(items):
        return '\n'.join(s.lstrip('/').strip() for s in items).strip()

    while i < n:
        line = lines[i].strip()

        if not line:
            pending_comment = []
            i += 1
            continue

        if line.startswith('//'):
            pending_comment.append(line)
            i += 1
            continue

        if line.startswith('#define'):
            # Single-line or backslash-continued macro.
            buf = [line]
            while buf[-1].rstrip().endswith('\\') and i + 1 < n:
                i += 1
                buf.append(lines[i])
            decls.append(Declaration(
                kind='define',
                body='\n'.join(buf),
                leading_comment=join_comment(pending_comment),
            ))
            pending_comment = []
            i += 1
            continue

        if line.startswith('typedef '):
            # Look for terminating ';'. If the typedef opens a brace, gobble
            # through the matching close brace first.
            buf = [line]
            depth = line.count('{') - line.count('}')
            while not (depth == 0 and buf[-1].rstrip().endswith(';')):
                i += 1
                if i >= n:
                    break
                buf.append(lines[i])
                depth += lines[i].count('{') - lines[i].count('}')
            body = '\n'.join(buf)

            # Classify.
            if re.search(r'typedef\s+enum\b', body):
                kind = 'enum'
            elif re.search(r'typedef\s+struct\s+\w+\s+\w+\s*;', body):
                # `typedef struct Foo Foo;` forward decl.
                kind = 'forward'
            elif re.search(r'typedef\s+struct\b.*\{', body, flags=re.DOTALL):
                kind = 'struct'
            else:
                kind = 'typedef'

            decls.append(Declaration(
                kind=kind,
                body=body,
                leading_comment=join_comment(pending_comment),
            ))
            pending_comment = []
            i += 1
            continue

        # Unknown / outside-our-scope line (forward fn decls inside
        # #ifndef blocks, etc.). Drop pending comments and move on.
        pending_comment = []
        i += 1

    return decls


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--header', default=None,
                        help='Path to remix_c.h (default: ../public/include/remix/remix_c.h relative to this script).')
    parser.add_argument('-o', '--output', default=None,
                        help='Path to write the generated markdown (default: ../RemixApiSurface.md).')
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    header_path = args.header or os.path.normpath(
        os.path.join(here, '..', 'public', 'include', 'remix', 'remix_c.h'))
    output_path = args.output or os.path.normpath(
        os.path.join(here, '..', 'RemixApiSurface.md'))

    with open(header_path, 'r', encoding='utf-8') as f:
        src = f.read()

    decls = tokenize_top_level(src)
    md = emit_full_document(decls, header_path=header_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Wrote {output_path} ({len(md)} bytes, {len(decls)} declarations).", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
