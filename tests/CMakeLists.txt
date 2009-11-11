set(categories
  reflexive
  regex
  zlib  
)

add_test(testsuite_all ${CMAKE_COMMAND} -P test_driver.cmake) 

foreach(category ${categories})
  add_test(testsuite_category_${category} ${CMAKE_COMMAND} -Dtest_category=${category} -P test_driver.cmake) 
endforeach(category)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_driver.cmake.in 
  ${CMAKE_CURRENT_BINARY_DIR}/test_driver.cmake
  @ONLY
)
