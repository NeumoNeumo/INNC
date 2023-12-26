doxygen_cmd = find_program('doxygen', required: false)
sphinx_cmd = find_program('sphinx-build', required: false)

doc_src = meson.current_source_dir()
doc_build = meson.current_build_dir()

# Doxygen configuration
doxygen_input_dir = doc_src / '../include'
doxygen_output_dir = doc_build / 'doxygen'

conf_data = configuration_data()
conf_data.set('PROJECT_NAME', meson.project_name())
conf_data.set('VERSION', meson.project_version())
conf_data.set('DOXYGEN_INPUT_DIR', doxygen_input_dir)
conf_data.set('DOXYGEN_OUTPUT_DIR', doxygen_output_dir)

doxyfile = configure_file(
  input: doc_src / 'Doxyfile.in',
  output: 'Doxyfile',
  configuration: conf_data
)

# Doxygen custom target
doxygen_target = custom_target(
  'Doxygen',
  input: doxyfile,
  output: ['doxygen',],
  command: [doxygen_cmd, '@INPUT@'],
  install: false,
  build_always_stale: true,
)


# Sphinx custom target
sphinx_target = custom_target(
  'Sphinx',
  input: files([
    doc_src / 'conf.py',
    doc_src / 'index.rst',
  ]),
  output: 'sphinx',
  command: [sphinx_cmd, '-b', 'html', doc_src, 
    '@OUTPUT@', '-Dbreathe_projects.INNC=@0@'.format(doxygen_output_dir / 'xml')
  ],
  install: false,
  depends: doxygen_target,
  build_always_stale: true,
)


