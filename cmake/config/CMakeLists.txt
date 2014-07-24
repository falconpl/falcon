#
# Configurator for development tools
#

# CMake will find it via find_package(falcon [REQUIRED])
# It supplies include paths, libraries (via falcon-core-targets.cmake)
# and convenience MACROS.
#

set( cmake_mod_files
   falcon-config.cmake
   falcon-config-version.cmake
   )
   
#configure also the internal, but don't ship it
foreach(item ${cmake_mod_files} falcon-config-internal.cmake )
  message( "Configuring ${item}" )
  configure_file(
    ${item}.in  "${CMAKE_CURRENT_BINARY_DIR}/${item}"
    ESCAPE_QUOTES @ONLY)

    #Create installation files from in files
    list( APPEND cmake_mod_files_inst ${CMAKE_CURRENT_BINARY_DIR}/${item} )
endforeach()

install(
   FILES ${cmake_mod_files_inst} 
   DESTINATION ${FALCON_CMAKE_DIR}
)
