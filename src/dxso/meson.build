dxso_src = files([
  'dxso_common.cpp',
  'dxso_options.cpp',
  'dxso_module.cpp',
  'dxso_reader.cpp',
  'dxso_header.cpp',
  'dxso_ctab.cpp',
  'dxso_util.cpp',
  'dxso_code.cpp',
  'dxso_tables.cpp',
  'dxso_decoder.cpp',
  'dxso_analysis.cpp',
  'dxso_preproj_position.cpp',
  'dxso_color_terms.cpp',
  'dxso_symbolic_eval.cpp',
  'dxso_highlight_tints.cpp',
  'dxso_material_fades.cpp',
  'dxso_opcode_util.cpp',
  'dxso_uv_dataflow.cpp',
  'dxso_sampler_inference.cpp',
  'dxso_ue3_material_identity.cpp',
  'dxso_compiler.cpp',
  'dxso_enums.cpp'
])

dxso_lib = static_library('dxso', dxso_src,
  include_directories : [ dxvk_include_path, dxvk_shader_include_path, usd_include_paths ],
  override_options    : ['cpp_std='+dxvk_cpp_std])

dxso_dep = declare_dependency(
  link_with           : [ dxso_lib ],
  include_directories : [ dxvk_include_path ])
