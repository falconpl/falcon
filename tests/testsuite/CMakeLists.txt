set(categories
  reflexive
  regex
  zlib  
)

add_test(testsuite_all ${CMAKE_INSTALL_PREFIX}/bin/faltest -d ${CMAKE_CURRENT_SOURCE_DIR} ) 

foreach(category ${categories})
  add_test(testsuite_category_${category} 
           ${CMAKE_INSTALL_PREFIX}/bin/faltest -d ${CMAKE_CURRENT_SOURCE_DIR} -c ${category}) 
endforeach(category)
