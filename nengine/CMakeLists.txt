##################################################
# Falcon Programming Language
#
##################################################

#Engine specific define
add_definitions(-DFALCON_ENGINE_EXPORTS)

include_directories(
  include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/core_module  
)

message(STATUS "Compiling Falcon core on ${CMAKE_SYSTEM_NAME}" )

#################
add_library(falcon_engine SHARED
  ../engine/autocstring.cpp
  ../engine/autoucsstring.cpp
  ../engine/autowstring.cpp
  ../engine/base64.cpp
  ../engine/basealloc.cpp
  compiler.cpp
  coreclass.cpp
  coreobject.cpp
  ../engine/fassert.cpp
  ../engine/filestat.cpp
  ../engine/format.cpp
  item.cpp
  symtab.cpp
  syntree.cpp
  src_lexer.cpp  
  vm.cpp
)

set_target_properties(falcon_engine PROPERTIES
  VERSION "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}"
  SOVERSION "${FALCON_SONAME_VERSION}"
)

#In unix and mac we have to add extra libraries
if(UNIX OR APPLE)
  #also dl for linux and solaris
  # and RT only for solaris and non-bsd systems
  set(SYS_LIBS pthread m)
  if(APPLE)
    list(APPEND SYS_LIBS dl)
  else()
    IF(CMAKE_SYSTEM_NAME MATCHES ".*BSD")
    else()
      list(APPEND SYS_LIBS dl rt)
    endif()
  endif()
  target_link_libraries(falcon_engine ${SYS_LIBS})
endif()

install(TARGETS falcon_engine 
  EXPORT falcon-core-targets
  ${FALCON_INSTALL_DESTINATIONS} 
)
