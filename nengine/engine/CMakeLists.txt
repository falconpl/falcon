##################################################
# Falcon Programming Language
#
##################################################

#Engine specific define
add_definitions(-DFALCON_ENGINE_EXPORTS)

message(STATUS "Compiling Falcon core on ${CMAKE_SYSTEM_NAME}" )

if(UNIX OR APPLE)
  set(SYS_SPECIFIC
    mt_posix.cpp
    sys_posix.cpp
  )
elseif(WIN32)

  set(SYS_SPECIFIC
    mt_win.cpp
    sys_win.cpp
  )
endif()

#################
add_library(falcon_engine SHARED
  ${SYS_SPECIFIC}

  memory.cpp
  application.cpp
  class.cpp
  closedsymbol.cpp
  codeerror.cpp
  collector.cpp
  corefunction.cpp
  corestring.cpp
  dynsymbol.cpp
  engine.cpp
  error.cpp
  errorclass.cpp
  fassert.cpp
  expression.cpp
  exprfactory.cpp
  exprsym.cpp
  exprvalue.cpp
  function.cpp
  genericerror.cpp
  globalsvector.cpp
  globalsymbol.cpp
  item.cpp
  localsymbol.cpp
  module.cpp
  operanderror.cpp
  pcode.cpp
  rampmode.cpp
  sourceref.cpp
  statement.cpp
  string.cpp
  stream.cpp
  symbol.cpp
  syntree.cpp
  trace.cpp
  tracestep.cpp
  vm.cpp
  vmcontext.cpp
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
