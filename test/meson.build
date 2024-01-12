gcovr_cmd = find_program('gcovr', required: false)
if not gcovr_cmd.found()
  message('`gcovr` not found. Coverage report will not generate.')
else
  root_dir = meson.source_root()
  gcovr_html_base = [gcovr_cmd, 
    '-r', root_dir, 
    '--exclude-throw-branches', 
    '--exclude-unreachable-branches', 
    '--merge-mode-functions', 'separate',
    '-e', root_dir / 'subprojects', 
    '-e', root_dir / 'test', 
    '--html', 
    '--html-details', 
    '--print-summary', 
    '-o', '@OUTPUT@']
  gcovr_xml_base = [gcovr_cmd, 
    '-r', root_dir,
    '--exclude-throw-branches', 
    '--exclude-unreachable-branches', 
    '--merge-mode-functions', 'separate',
    '-e', root_dir /'subprojects', 
    '-e', root_dir / 'test', 
    '-x', 
    '-o', '@OUTPUT@']
  cpp_compiler = meson.get_compiler('cpp')
  if cpp_compiler.get_id() == 'clang'
    gcovr_html_base += ['--gcov-executable', 'llvm-cov gcov']
    gcovr_xml_base += ['--gcov-executable', 'llvm-cov gcov']
  endif
  custom_target(
    'coverage_excl_html_target',
    output: 'coverage.html',
    command: gcovr_html_base,
    install: false,
    build_always_stale: true,
  )
  custom_target(
    'coverage_excl_xml_target',
    output: 'coverage.xml',
    command: gcovr_xml_base,
    install: false,
    build_always_stale: true,
  )
endif
