gcovr_cmd = find_program('gcovr', required: false)
if not gcovr_cmd.found()
  message('`gcovr` not found. Coverage report will not generate.')
else
  root_dir = meson.source_root()
  custom_target(
    'coverage_excl_html_target',
    output: 'coverage.html',
    command: [gcovr_cmd, '-r', root_dir, '--exclude-throw-branches', 
      '--exclude-unreachable-branches', '--html', '--html-details', 
      '--print-summary', '-e', root_dir / 'subprojects', '-e',
      root_dir / 'test', '-o',
      '@OUTPUT@'
    ],
    install: false,
    build_always_stale: true,
  )
  custom_target(
    'coverage_excl_xml_target',
    output: 'coverage.xml',
    command: [gcovr_cmd, '-r', root_dir, '--exclude-throw-branches', 
      '--exclude-unreachable-branches', '-x', '-e', root_dir /'subprojects', 
      '-e', root_dir / 'test', '-o', '@OUTPUT@'
    ],
    install: false,
    build_always_stale: true,
  )
endif
