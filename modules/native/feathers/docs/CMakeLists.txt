find_package(faldoc REQUIRED)

set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/Falcon-feathers-docs.${FALCON_VERSION_ID})
#
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/faldoc.fd.in
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd
  @ONLY
)
#
faldoc_module_docs(feathers
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd 
  ${output_dir}
)
