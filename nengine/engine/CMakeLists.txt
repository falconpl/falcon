##################################################
# Falcon Programming Language
#
##################################################

#Engine specific define
add_definitions(-DFALCON_ENGINE_EXPORTS)

message(STATUS "Compiling Falcon core on ${CMAKE_SYSTEM_NAME}" )

if(UNIX OR APPLE)
  set(SYS_SPECIFIC
    fstream_posix.cpp
    interrupt_posix.cpp
    mt_posix.cpp
    stdstreams_posix.cpp
    sys_posix.cpp
    timestamp_posix.cpp
    vfs_file_posix.cpp
  )
elseif(WIN32)

  set(SYS_SPECIFIC
    fstream_win.cpp
    interrupt_win.cpp
    mt_win.cpp
    stdstreams_win.cpp
    sys_win.cpp
    timestamp_win.cpp
    vfs_file_win.cpp
  )
endif()

#################
add_library(falcon_engine SHARED
  ${SYS_SPECIFIC}

  parser/lexer.cpp
  parser/literal.cpp
  parser/nonterminal.cpp
  parser/parser.cpp
  parser/predef.cpp
  parser/token.cpp
  parser/tokeninstance.cpp

  memory.cpp
  application.cpp
  autocstring.cpp
  autowstring.cpp
  class.cpp
  closedsymbol.cpp
  codeerror.cpp
  collector.cpp
  corefunction.cpp
  corenil.cpp
  coreint.cpp
  corestring.cpp
  datareader.cpp
  datawriter.cpp
  directory.cpp
  dynsymbol.cpp
  encodingerror.cpp
  engine.cpp
  error.cpp
  errorclass.cpp
  extfunc.cpp
  fassert.cpp
  fatal.cpp
  fstream.cpp
  expression.cpp
  exprfactory.cpp
  exprsym.cpp
  exprvalue.cpp
  filestat.cpp
  function.cpp
  genericerror.cpp
  globalsvector.cpp
  globalsymbol.cpp
  item.cpp
  interruptederror.cpp
  ioerror.cpp
  localsymbol.cpp
  module.cpp
  operanderror.cpp
  path.cpp
  pcode.cpp
  rampmode.cpp
  reader.cpp
  rulesyntree.cpp
  sourceref.cpp
  statement.cpp
  stmtrule.cpp
  string.cpp
  stream.cpp
  symbol.cpp
  synfunc.cpp
  syntree.cpp
  textreader.cpp
  textwriter.cpp
  timestamp.cpp
  transcoder.cpp
  transcoderc.cpp
  transcoderutf8.cpp
  trace.cpp
  tracestep.cpp
  unsupportederror.cpp
  uri.cpp
  vfsiface.cpp
  vfsprovider.cpp
  vm.cpp
  vmcontext.cpp
  writer.cpp

  parser/token.cpp
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
