find_program(Falcon_faldoc_EXECUTABLE faldoc.sh "${Falcon_BINARY_DIR}")
if(NOT Falcon_faldoc_EXECUTABLE )
  message(FATAL_ERROR "faldoc not found")
endif()

# WARNING, FIXME: faldoc isn't able to generate into the build dir right now.
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
  COMMAND ${Falcon_faldoc_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd
) 
add_custom_target(create_docs ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/index.html)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ DESTINATION share/doc/falcon/modules/hpdf/html)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/faldoc.fd.in
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd
  @ONLY
)
