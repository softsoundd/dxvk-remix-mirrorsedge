lssUsd_src = files([
  'game_exporter.cpp',
  'usd_mesh_importer.cpp',
  'usd_mesh_importer.h',
  'usd_mesh_samplers.h',
  'hd/usd_mesh_util.cpp',
  'hd/usd_mesh_util.h',
  'mdl_helpers.h'
])

if not dxvk_is_msvc
  lss_Usd_args = ['-fpermissive']
else
  lss_Usd_args = []
endif
thread_dep = dependency('threads')

# The fork's own MDL, so captured materials expose the parameters only this runtime reads, such as the texture transform
# and the flow layer.
libusd_embedded_data_files = files([
  '../mdl/Base/AperturePBR_Opacity.mdl',
  '../mdl/Base/AperturePBR_Translucent.mdl',
  '../mdl/Base/AperturePBR_Model.mdl',
  '../mdl/Base/AperturePBR_Normal.mdl',
  '../mdl/Base/AperturePBR_SpriteSheet.mdl',
])

libusd_generated_embedded_files = embedding_compiler.process(libusd_embedded_data_files)

lssUsd_lib = static_library('lssUsd', lssUsd_src,[libusd_generated_embedded_files],
  dependencies        : [ usd_dep, embedded_dep ],
  include_directories : [ usd_include_paths, '../../include', vk_include_path ],
  override_options    : ['cpp_std='+dxvk_cpp_std],
  cpp_args            : lss_Usd_args)

lssUsd_dep = declare_dependency(
  link_with           : [ lssUsd_lib ])
