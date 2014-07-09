# Maybe useful for C++ application and module writers, thus a reusable function
function(generate_doxygen_documentation _target_name _doxygen_config_file)
  find_package(Doxygen)
  if(NOT DOXYGEN_FOUND )
    message(STATUS "Doxygen not found - Doxygen based docs won't be generated")
    return()
  endif()

  if(NOT EXISTS ${_doxygen_config_file})
    message( STATUS "Doxygen configuration file not found - Doxygen based docs won't be generated")
    return()
  endif()

  add_custom_target(${_target_name}
    COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYGEN_CONFIG_FILE}"
  )
endfunction(generate_doxygen_documentation)

set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/Falcon-docs-engine)
configure_file(
  Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  @ONLY
)
#
generate_doxygen_documentation(
  engine_docs # make target
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
)

#
install(
   DIRECTORY ${output_dir} DESTINATION "${FALCON_DOC_DIR}"
   OPTIONAL # if docs haven't been generated
)

message(STATUS "Configuring faldoc.fd")
configure_file(
  faldoc.fd.in # input
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.fd
  @ONLY
)

if(WIN32)
    option( FALCON_WITH_MANPAGES "Build and install also manpages" OFF )
else()
    # Manpages are mandatory on unix
    option( FALCON_WITH_MANPAGES "Build and install also manpages" ON )
endif()
#
if (FALCON_WITH_MANPAGES)
   add_subdirectory(manpages)
endif()
