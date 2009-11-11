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

add_test(testsuite_all ${CMAKE_INSTALL_PREFIX}/bin/faltest -d ${CMAKE_CURRENT_BINARY_DIR} ) 

foreach(category ${categories})
  add_test(testsuite_category_${category} 
           ${CMAKE_INSTALL_PREFIX}/bin/faltest -d ${CMAKE_CURRENT_SOURCE_DIR} -c ${category}) 
endforeach(category)

