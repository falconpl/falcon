file(GLOB _headers *.h)
   
unset(WOPI_HEADERS)
foreach(item ${_headers})
  list(APPEND WOPI_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${item})
endforeach(item)
set(WOPI_HEADERS ${WOPI_HEADERS} PARENT_SCOPE)

install(FILES ${WOPI_HEADERS} DESTINATION ${Falcon_INC_DIR}/falcon/wopi)