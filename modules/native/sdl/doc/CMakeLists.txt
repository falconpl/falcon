find_package(faldoc REQUIRED)

set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/html)
#
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/faldoc.fd.in
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd
  @ONLY
)
#
faldoc_module_docs(sdl
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd 
  ${output_dir}
)

#install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ DESTINATION share/doc/falcon/modules/sdl-0.9.12/html)
