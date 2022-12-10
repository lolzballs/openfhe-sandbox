project('openfhe-sandbox', 'cpp',
  version: '0.1',
  default_options: ['warning_level=3', 'cpp_std=c++17'])

add_project_arguments('-DMATHBACKEND=4', language: 'cpp')

cmake = import('cmake')

openfhe_options = cmake.subproject_options()
openfhe_options.add_cmake_defines({
  'BUILD_UNITTESTS': false,
  'BUILD_EXAMPLES': false,
  'BUILD_BENCHMARKS': false,
  'BUILD_OPENMP': false,
  'MATHBACKEND': 4,
})
openfhe_subproj = cmake.subproject('openfhe', options: openfhe_options)

dep_openfhe_core = openfhe_subproj.dependency('OPENFHEcore', include_type: 'system')
dep_openfhe_pke = openfhe_subproj.dependency('OPENFHEpke', include_type: 'system')

includes = include_directories('include')

executable(
  'openfhe_sandbox',
  'src/openfhe_sandbox.cpp',
  dependencies: [dep_openfhe_core, dep_openfhe_pke],
  install: true
)

