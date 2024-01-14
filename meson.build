project('INNC', 'cpp',
  default_options : ['cpp_std=c++20'],
  version : '0.1.0',
  license : 'MIT')

incdir = include_directories('include')

gtest_proj = subproject('gtest')
gtest_dep = gtest_proj.get_variable('gtest_main_dep')
gmock_dep = gtest_proj.get_variable('gmock_dep')

subdir('docs')
subdir('test')

libINNC = library('INNC',
  'src/INNC/function.cpp',
  'src/INNC/storage.cpp',
  'src/INNC/tensor.cpp',
  'src/INNC/tensorImpl.cpp',
  'src/INNC/types.cpp',
  'src/INNC/layouts.cpp',
  'src/INNC/utils/utils.cpp',
  'src/INNC/utils/rand.cpp',
  include_directories: incdir,
)

e = executable('unittest', 
  'test/test.cpp', 
  link_with: [libINNC, ],
  include_directories: incdir,
  dependencies: [
    gtest_dep,
    gmock_dep
  ]
)

test('Test', e)

