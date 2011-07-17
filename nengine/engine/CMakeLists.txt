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
  parser/rule.cpp
  parser/state.cpp
  parser/token.cpp
  parser/tokeninstance.cpp

  core_module/compare.cpp
  core_module/describe.cpp
  core_module/len.cpp
  core_module/minmax.cpp
  core_module/print.cpp
  core_module/tostring.cpp
  core_module/typeid.cpp
  core_module/coremodule.cpp

  sp/parser_arraydecl.cpp
  sp/parser_assign.cpp
  sp/parser_atom.cpp
  sp/parser_autoexpr.cpp
  sp/parser_call.cpp
  sp/parser_class.cpp
  sp/parser_deletor.cpp
  sp/parser_end.cpp
  sp/parser_expr.cpp
  sp/parser_function.cpp
  sp/parser_if.cpp
  sp/parser_index.cpp
  sp/parser_list.cpp
  sp/parser_rule.cpp
  sp/parser_while.cpp
  sp/parsercontext.cpp
  sp/sourcelexer.cpp
  sp/sourceparser.cpp

  memory.cpp
  accesserror.cpp
  accesstypeerror.cpp
  application.cpp
  autocstring.cpp
  autowstring.cpp
  bom.cpp
  class.cpp
  classarray.cpp
  classbool.cpp
  classdict.cpp
  classfunction.cpp
  classint.cpp
  classnil.cpp
  classnumeric.cpp
  classstring.cpp
  closedsymbol.cpp
  codeerror.cpp
  collector.cpp
  collector_history.cpp
  datareader.cpp
  datawriter.cpp
  directory.cpp
  dynsymbol.cpp
  encodingerror.cpp
  engine.cpp
  error.cpp
  errorclass.cpp
  exprarray.cpp
  exprcall.cpp
  exprcompare.cpp
  exprdict.cpp
  exprunpack.cpp
  exprmultiunpack.cpp
  extfunc.cpp
  fassert.cpp
  falconclass.cpp
  falconinstance.cpp
  falconstate.cpp
  fatal.cpp
  fstream.cpp
  expression.cpp
  exprfactory.cpp
  exprmath.cpp
  exprsym.cpp
  exprvalue.cpp
  filestat.cpp
  function.cpp
  genericerror.cpp
  globalsymbol.cpp
  hyperclass.cpp
  inheritance.cpp
  intcompiler.cpp
  item.cpp
  item_util.cpp
  itemarray.cpp
  interruptederror.cpp
  ioerror.cpp
  localsymbol.cpp
  metaclass.cpp
  module.cpp
  operanderror.cpp
  overridableclass.cpp
  paramerror.cpp
  pseudofunc.cpp
  path.cpp
  pcode.cpp
  rampmode.cpp
  reader.cpp
  rulesyntree.cpp
  sourceref.cpp
  statement.cpp
  stmt_init.cpp
  stmtrule.cpp
  string.cpp
  stringstream.cpp
  stream.cpp
  symbol.cpp
  symboltable.cpp
  synfunc.cpp
  syntaxerror.cpp
  syntree.cpp
  textreader.cpp
  textwriter.cpp
  timestamp.cpp
  transcoder.cpp
  transcoderc.cpp
  transcoderutf8.cpp
  trace.cpp
  tracestep.cpp
  unknownsymbol.cpp
  unsupportederror.cpp
  uri.cpp
  vfsiface.cpp
  vfsprovider.cpp
  vm.cpp
  vmcontext.cpp
  writer.cpp
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
