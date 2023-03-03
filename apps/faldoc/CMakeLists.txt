####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

cmake_minimum_required(VERSION 3.16.3)
project(falcon-app-faldoc)

set(FALDOC_DIR "${FALCON_APP_DIR}/faldoc" )

#install startup bat

if(UNIX)
   configure_file(
   faldoc.sh.in
   ${CMAKE_CURRENT_BINARY_DIR}/faldoc
   @ONLY
   )

   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/faldoc DESTINATION 
${FALCON_BIN_DIR}
   PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
   )
else()
   configure_file(
   faldoc.bat.in
   ${CMAKE_CURRENT_BINARY_DIR}/faldoc.bat
   @ONLY
   )

   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/faldoc.bat DESTINATION 
${FALCON_BIN_DIR} )
endif()

install(FILES src/faldoc.fal DESTINATION ${FALDOC_DIR})
install(DIRECTORY src/faldoc DESTINATION ${FALDOC_DIR})
install(DIRECTORY src/resources DESTINATION ${FALDOC_DIR})

configure_file(
  faldoc-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc-config.cmake
  @ONLY
)       

install(
   FILES ${CMAKE_CURRENT_BINARY_DIR}/faldoc-config.cmake
   DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cmake/faldoc
)
