project('libtensor', 'cpp',
  default_options : ['cpp_std=c++17'],
  version : '0.1.0',
  license : 'MIT')

incdir = include_directories('include')

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_main_dep')
gmock_dep = gtest_proj.get_variable('gmock_dep')

e = executable('unittest', 
  'test/test.cpp', 
  include_directories: incdir,
  dependencies: [
    gtest_dep,
    gmock_dep
  ]
)

test('Test', e)

