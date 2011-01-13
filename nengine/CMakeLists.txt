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
  ../engine/basealloc.cpp  
  ../engine/fassert.cpp
  ../engine/string.cpp
  ../engine/stream.cpp
  ../engine/memory.cpp
  ../engine/falcondata.cpp
  ../engine/gcalloc.cpp  
  
  closedsymbol.cpp
  dynsymbol.cpp
  expression.cpp
  exprfactory.cpp
  exprvalue.cpp
  garbagelock.cpp
  globalsymbol.cpp
  item.cpp
  localsymbol.cpp
  pcode.cpp
  sourceref.cpp
  statement.cpp
  symbol.cpp
  syntree.cpp
  vm.cpp
)

add_executable( test
	main.cpp )
	
target_link_libraries( test
	falcon_engine m )

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
