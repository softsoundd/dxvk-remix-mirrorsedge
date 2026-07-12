"""Unit tests for generate_remix_api_md."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generate_remix_api_md as gen


class TestTokenizer(unittest.TestCase):
    def test_splits_top_level_declarations(self):
        source = """
#define FOO 1
#define BAR 2

typedef uint32_t my_bool;

typedef enum my_enum {
  MY_ENUM_A = 0,
  MY_ENUM_B = 1,
} my_enum;

typedef struct my_struct {
  int32_t x;
  int32_t y;
} my_struct;
"""
        decls = gen.tokenize_top_level(source)
        kinds = [d.kind for d in decls]
        self.assertEqual(kinds, ['define', 'define', 'typedef', 'enum', 'struct'])

    def test_strips_block_comments_but_preserves_doc_lines(self):
        source = """
/* a multi-line
   block comment that should be stripped */
// doc comment for FOO
#define FOO 1
"""
        decls = gen.tokenize_top_level(source)
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0].kind, 'define')
        self.assertIn('doc comment for FOO', decls[0].leading_comment)


class TestParseDefine(unittest.TestCase):
    def test_simple_constant(self):
        decl = gen.Declaration(kind='define', body='#define FOO 42')
        result = gen.parse_define(decl)
        self.assertEqual(result.name, 'FOO')
        self.assertEqual(result.value, '42')

    def test_function_like_macro(self):
        decl = gen.Declaration(
            kind='define',
            body='#define MAKE_VER(a, b, c) ( (a) << 16 | (b) << 8 | (c) )',
        )
        result = gen.parse_define(decl)
        self.assertEqual(result.name, 'MAKE_VER')
        self.assertEqual(result.params, ['a', 'b', 'c'])
        self.assertIn('<< 16', result.value)

    def test_multiline_continuation(self):
        body = '#define X (1 \\\n    | 2 \\\n    | 4)'
        decl = gen.Declaration(kind='define', body=body)
        result = gen.parse_define(decl)
        self.assertEqual(result.name, 'X')
        self.assertIn('| 2', result.value)
        self.assertIn('| 4', result.value)


class TestParseTypedef(unittest.TestCase):
    def test_simple_scalar_typedef(self):
        decl = gen.Declaration(kind='typedef', body='typedef uint32_t remixapi_Bool;')
        result = gen.parse_simple_typedef(decl)
        self.assertEqual(result.name, 'remixapi_Bool')
        self.assertEqual(result.aliased_type, 'uint32_t')

    def test_handle_typedef(self):
        decl = gen.Declaration(
            kind='typedef',
            body='typedef struct remixapi_MaterialHandle_T* remixapi_MaterialHandle;',
        )
        result = gen.parse_simple_typedef(decl)
        self.assertEqual(result.name, 'remixapi_MaterialHandle')
        self.assertEqual(result.aliased_type, 'struct remixapi_MaterialHandle_T*')

    def test_forward_struct_decl(self):
        decl = gen.Declaration(
            kind='forward',
            body='typedef struct IDirect3D9Ex IDirect3D9Ex;',
        )
        result = gen.parse_simple_typedef(decl)
        self.assertEqual(result.name, 'IDirect3D9Ex')
        self.assertEqual(result.aliased_type, 'struct IDirect3D9Ex')


class TestParseFnPtrTypedef(unittest.TestCase):
    def test_simple_fn_ptr(self):
        body = (
            'typedef remixapi_ErrorCode (REMIXAPI_PTR* PFN_remixapi_Shutdown)(void);'
        )
        decl = gen.Declaration(kind='typedef', body=body)
        result = gen.parse_fn_ptr_typedef(decl)
        self.assertEqual(result.name, 'PFN_remixapi_Shutdown')
        self.assertEqual(result.return_type, 'remixapi_ErrorCode')
        self.assertEqual(result.params, ['void'])

    def test_multi_param_fn_ptr(self):
        body = (
            'typedef remixapi_ErrorCode (REMIXAPI_PTR* PFN_remixapi_CreateMaterial)('
            '  const remixapi_MaterialInfo* info,'
            '  remixapi_MaterialHandle*     out_handle);'
        )
        decl = gen.Declaration(kind='typedef', body=body)
        result = gen.parse_fn_ptr_typedef(decl)
        self.assertEqual(result.name, 'PFN_remixapi_CreateMaterial')
        self.assertEqual(result.return_type, 'remixapi_ErrorCode')
        self.assertEqual(len(result.params), 2)
        self.assertIn('const remixapi_MaterialInfo*', result.params[0])
        self.assertIn('remixapi_MaterialHandle*', result.params[1])

    def test_returns_none_for_non_fn_ptr(self):
        decl = gen.Declaration(
            kind='typedef',
            body='typedef uint32_t remixapi_Bool;',
        )
        self.assertIsNone(gen.parse_fn_ptr_typedef(decl))


class TestParseEnum(unittest.TestCase):
    def test_basic_enum(self):
        body = """
typedef enum remixapi_ErrorCode {
  REMIXAPI_ERROR_CODE_SUCCESS         = 0,
  REMIXAPI_ERROR_CODE_GENERAL_FAILURE = 1,
} remixapi_ErrorCode;
"""
        decl = gen.Declaration(kind='enum', body=body)
        result = gen.parse_enum(decl)
        self.assertEqual(result.name, 'remixapi_ErrorCode')
        self.assertEqual(len(result.members), 2)
        self.assertEqual(result.members[0].name, 'REMIXAPI_ERROR_CODE_SUCCESS')
        self.assertEqual(result.members[0].value, '0')
        self.assertEqual(result.members[1].name, 'REMIXAPI_ERROR_CODE_GENERAL_FAILURE')
        self.assertEqual(result.members[1].value, '1')

    def test_enum_with_hex_values_and_member_comments(self):
        body = """
typedef enum remixapi_X {
  // success!
  REMIXAPI_X_OK                                = 0,
  REMIXAPI_X_HRESULT_NO_REQUIRED_GPU_FEATURES  = 0x88960001,
} remixapi_X;
"""
        decl = gen.Declaration(kind='enum', body=body)
        result = gen.parse_enum(decl)
        self.assertEqual(result.members[0].value, '0')
        self.assertEqual(result.members[0].comment, 'success!')
        self.assertEqual(result.members[1].value, '0x88960001')

    def test_enum_with_implicit_values(self):
        body = """
typedef enum remixapi_Y {
  REMIXAPI_Y_FIRST,
  REMIXAPI_Y_SECOND,
} remixapi_Y;
"""
        decl = gen.Declaration(kind='enum', body=body)
        result = gen.parse_enum(decl)
        self.assertEqual(result.members[0].value, '(implicit)')
        self.assertEqual(result.members[1].value, '(implicit)')


class TestParseStruct(unittest.TestCase):
    def test_basic_struct(self):
        body = """
typedef struct remixapi_Float3D {
  float x;
  float y;
  float z;
} remixapi_Float3D;
"""
        decl = gen.Declaration(kind='struct', body=body)
        result = gen.parse_struct(decl)
        self.assertEqual(result.name, 'remixapi_Float3D')
        self.assertEqual([f.name for f in result.fields], ['x', 'y', 'z'])
        self.assertEqual(result.fields[0].type, 'float')

    def test_struct_with_stype_and_pnext(self):
        body = """
typedef struct remixapi_MaterialInfo {
  remixapi_StructType sType;
  void*               pNext;
  uint64_t            hash;
  const wchar_t*      albedoTexture;
} remixapi_MaterialInfo;
"""
        decl = gen.Declaration(kind='struct', body=body)
        result = gen.parse_struct(decl)
        self.assertEqual(result.name, 'remixapi_MaterialInfo')
        self.assertTrue(result.is_info_struct)
        self.assertEqual(result.fields[0].name, 'sType')
        self.assertEqual(result.fields[3].name, 'albedoTexture')
        self.assertEqual(result.fields[3].type, 'const wchar_t*')

    def test_struct_with_field_comment(self):
        body = """
typedef struct remixapi_Foo {
  // this is the x coordinate
  float x;
} remixapi_Foo;
"""
        decl = gen.Declaration(kind='struct', body=body)
        result = gen.parse_struct(decl)
        self.assertEqual(result.fields[0].comment, 'this is the x coordinate')

    def test_interface_struct_keeps_pfn_fields(self):
        body = """
typedef struct remixapi_Interface {
  PFN_remixapi_Shutdown        Shutdown;
  PFN_remixapi_CreateMaterial  CreateMaterial;
  // DXVK interoperability
  PFN_remixapi_dxvk_CreateD3D9 dxvk_CreateD3D9;
} remixapi_Interface;
"""
        decl = gen.Declaration(kind='struct', body=body)
        result = gen.parse_struct(decl)
        self.assertEqual(result.name, 'remixapi_Interface')
        self.assertEqual(len(result.fields), 3)
        self.assertEqual(result.fields[0].type, 'PFN_remixapi_Shutdown')
        self.assertEqual(result.fields[2].comment, 'DXVK interoperability')


class TestEmit(unittest.TestCase):
    def test_emit_defines_section(self):
        defines = [
            gen.Define(name='REMIXAPI_VERSION_MAJOR', value='0'),
            gen.Define(name='REMIXAPI_VERSION_MAKE',
                       value='( (a) << 48 )',
                       params=['major', 'minor', 'patch']),
        ]
        md = gen.emit_defines_section(defines)
        self.assertIn('## Macros and version', md)
        self.assertIn('`REMIXAPI_VERSION_MAJOR`', md)
        self.assertIn('`0`', md)
        self.assertIn('REMIXAPI_VERSION_MAKE(major, minor, patch)', md)

    def test_emit_enum_section(self):
        e = gen.Enum(name='remixapi_ErrorCode', members=[
            gen.EnumMember(name='REMIXAPI_ERROR_CODE_SUCCESS', value='0'),
            gen.EnumMember(name='REMIXAPI_ERROR_CODE_GENERAL_FAILURE', value='1'),
        ])
        md = gen.emit_enum(e)
        self.assertIn('### `remixapi_ErrorCode`', md)
        self.assertIn('REMIXAPI_ERROR_CODE_SUCCESS', md)
        self.assertIn('| `0` |', md)

    def test_emit_struct_section(self):
        s = gen.Struct(name='remixapi_Float3D', fields=[
            gen.StructField(name='x', type='float'),
            gen.StructField(name='y', type='float'),
            gen.StructField(name='z', type='float'),
        ])
        md = gen.emit_struct(s)
        self.assertIn('### `remixapi_Float3D`', md)
        self.assertIn('| `float` | `x` |', md)

    def test_emit_fn_ptr(self):
        f = gen.FnPtrTypedef(
            name='PFN_remixapi_Shutdown',
            return_type='remixapi_ErrorCode',
            params=['void'],
        )
        md = gen.emit_fn_ptr(f)
        self.assertIn('### `PFN_remixapi_Shutdown`', md)
        self.assertIn('Returns: `remixapi_ErrorCode`', md)


class TestFullRun(unittest.TestCase):
    def test_generator_produces_known_elements_against_real_header(self):
        """Smoke test: run the full generator against remix_c.h and check
        that known landmarks appear in the output."""
        import io
        here = os.path.dirname(os.path.abspath(__file__))
        header = os.path.normpath(os.path.join(
            here, '..', 'public', 'include', 'remix', 'remix_c.h'))
        if not os.path.exists(header):
            self.skipTest('remix_c.h not found at expected path')
        with open(header, 'r', encoding='utf-8') as f:
            src = f.read()
        decls = gen.tokenize_top_level(src)
        md = gen.emit_full_document(decls, header_path=header)
        self.assertIn('REMIXAPI_VERSION_MAJOR', md)
        self.assertIn('remixapi_ErrorCode', md)
        self.assertIn('remixapi_Interface', md)
        self.assertIn('PFN_remixapi_Shutdown', md)
        self.assertIn('remixapi_MaterialInfo', md)


if __name__ == '__main__':
    unittest.main()
