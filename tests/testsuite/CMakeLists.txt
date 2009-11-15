set(categories
  smoke
  types
  comp
  functional
  messages
  basic
  expressions
  classes
  function
  lambda
  expression
  switch
  rtl
  RTL
  statements
  loops
  fbom
  tabular
  functions
  gc
  modloader
  iterators
  macro
  membuf
  meta
  methods
  poop
  prototype
  reference
  select
  template
)

# clean this directory from artefacts of former faltest runs
file(GLOB _fams "*.fam") 
if(_fams)
  file(REMOVE ${_fams})
endif(_fams)

# start memcheck ignore list 
file(WRITE ${CMAKE_BINARY_DIR}/CTestCustom.cmake "set(CTEST_CUSTOM_MEMCHECK_IGNORE\n")

add_test(testsuite_all ${CMAKE_COMMAND} -P test_driver.cmake) 
file(APPEND ${CMAKE_BINARY_DIR}/CTestCustom.cmake "  testsuite_all\n")   

foreach(category ${categories})
  add_test(testsuite_category_${category} ${CMAKE_COMMAND} -Dtest_category=${category} -P test_driver.cmake) 
  file(APPEND ${CMAKE_BINARY_DIR}/CTestCustom.cmake "  testsuite_category_${category}\n")   
endforeach(category)

# wrap memcheck ignore list up
file(APPEND ${CMAKE_BINARY_DIR}/CTestCustom.cmake ")")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_driver.cmake.in 
  ${CMAKE_CURRENT_BINARY_DIR}/test_driver.cmake
  @ONLY
)
