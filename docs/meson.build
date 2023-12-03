doxygen_cmd = find_program('doxygen', required: true)
sphinx_cmd = find_program('sphinx-build', required: true)

# Find header

# Doxygen configuration
doxygen_input_dir = meson.current_source_dir() / '../include'
doxygen_output_dir = meson.current_build_dir() / 'doxygen'
doxygen_generated = 'doxygen'
doxyfile_in = meson.current_source_dir() / 'Doxyfile.in'
doxyfile_out = 'Doxyfile'
doxyfile_out_path = meson.current_build_dir() / doxyfile_out

conf_data = configuration_data()
conf_data.set('DOXYGEN_INPUT_DIR', doxygen_input_dir)
conf_data.set('DOXYGEN_OUTPUT_DIR', doxygen_output_dir)
conf_data.set('VERSION', meson.project_version())

doxyfile = configure_file(
  input: doxyfile_in,
  output: doxyfile_out,
  configuration: conf_data
)

# Doxygen custom target
doxygen_target = custom_target(
  'Doxygen',
  input: doxyfile,
  output: doxygen_generated,
  command: [doxygen_cmd, doxyfile],
  install: false,
)

## Sphinx configuration
sphinx_source = meson.current_source_dir()
sphinx_build = meson.current_build_dir() / 'sphinx'

# Sphinx custom target
sphinx_target = custom_target(
  'Sphinx',
  input: files([
    sphinx_source / 'conf.py',
    meson.current_source_dir() / 'index.rst',
  ]),
  output: 'sphinx',
  command: [sphinx_cmd, '-b', 'html', sphinx_source, 
    sphinx_build, '-Dbreathe_projects.libtensor=@0@'.format(doxygen_output_dir / 'xml')
  ],
  install: false,
  depends: doxygen_target
)


