subdir('unit')

dxvkrt_test_root = meson.global_source_root().replace('\\', '/') + '/tests/rtx/dxvk_rt_testing/'
# softsoundd note - some checkouts may contain the `dxvk_rt_testing` runtime assets directory without the
# corresponding meson build scripts, only include the subdir when a `meson.build` is present
if fs.is_file('dxvk_rt_testing/meson.build')
  subdir('dxvk_rt_testing')
endif

subdir('apps/RemixAPI')
subdir('apps/RemixAPI_C')
subdir('apps/HydraTestRender')
if dxvk_is_ninja
  # apps that are compiled as a part of dxvk-remix
  dxvkrt_output_targets += {
    'apics/RemixAPI'    : RemixAPI_exepath,
    'apics/RemixAPI_C'  : RemixAPI_C_exepath,
    'apics/HydraTestRender'  : HydraTestRender_exepath,
  }
endif