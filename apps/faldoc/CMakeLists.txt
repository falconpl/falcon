####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

cmake_minimum_required(VERSION 2.6.2)
project(falcon-app-faldoc)

find_package(Falcon REQUIRED)

if(Falcon_IN_CORE_SOURCETREE)
  if(WIN32)
    set(${CMAKE_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}/falcon  -S -P faldoc ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc.fal)
  else()
    set(_cmd
      ${CMAKE_COMMAND}
          -Dexecutable=falpack
          -Dworkdir=${CMAKE_CURRENT_BINARY_DIR}
          -Darguments="-S -P faldoc ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc.fal"
        -P ${CMAKE_BINARY_DIR}/${FALCON_BIN_DIR}/executable_wrapper.cmake
    )
  endif()
else()
  set(_cmd ${Falcon_falpack_EXECUTABLE} -S -P faldoc ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc.fal)
endif()

add_custom_command(
  OUTPUT faldoc/faldoc.fal
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc.fal
  COMMAND ${_cmd}
)

add_custom_target(faldoc-package ALL DEPENDS faldoc/faldoc.fal)
if(Falcon_IN_CORE_SOURCETREE)
  add_dependencies(faldoc-package falcon Falcon_Feathers)
endif()

#workaround for falpack shortcoming
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/faldoc/output
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/faldoc
)

configure_file(
  faldoc.sh.in
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


configure_file(
  faldoc-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/faldoc-config.cmake
  @ONLY
)

install(
   FILES ${CMAKE_CURRENT_BINARY_DIR}/faldoc-config.cmake
   DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cmake/faldoc
)
