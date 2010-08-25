####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

cmake_minimum_required(VERSION 2.8)
project(falcon-module-hpdf)

find_package(Falcon REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/faldoc/faldoc.sh
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc.fal
  COMMAND ${Falcon_falpack_EXECUTABLE} -S -P faldoc ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc.fal 
) 
add_custom_target(create_fal-package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/faldoc/faldoc.sh)
#workaround for falpack shortcoming
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc/output 
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/faldoc/faldoc
  PATTERN ".svn" EXCLUDE
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/faldoc.sh.in 
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc.sh
  @ONLY
)            
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/faldoc DESTINATION ${Falcon_APP_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/faldoc.sh DESTINATION ${Falcon_BIN_DIR} 
  PERMISSIONS 
    OWNER_READ OWNER_WRITE OWNER_EXECUTE                                             							 
    GROUP_READ GROUP_EXECUTE                    							 
    WORLD_READ WORLD_EXECUTE 
)

