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
   
  if(APPLE)
    set(SYS_SPECIFIC
       ${SYS_SPECIFIC}
       dynloader_macosx.cpp
    )
  else()
    set(SYS_SPECIFIC
       ${SYS_SPECIFIC}
       dynloader_posix.cpp
    )
  endif()

elseif(WIN32)

  set(SYS_SPECIFIC
    dynloader_win.cpp
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
  core_module/path.cpp
  core_module/print.cpp
  core_module/tostring.cpp
  core_module/typeid.cpp
  core_module/clone.cpp
  core_module/coremodule.cpp
  core_module/uri.cpp

  sp/parser_arraydecl.cpp
  sp/parser_assign.cpp
  sp/parser_atom.cpp
  sp/parser_autoexpr.cpp
  sp/parser_call.cpp
  sp/parser_class.cpp
  sp/parser_deletor.cpp
  sp/parser_end.cpp
  sp/parser_export.cpp
  sp/parser_expr.cpp
  sp/parser_fastprint.cpp
  sp/parser_for.cpp
  sp/parser_function.cpp
  sp/parser_if.cpp
  sp/parser_import.cpp
  sp/parser_index.cpp
  sp/parser_list.cpp
  sp/parser_load.cpp
  sp/parser_namespace.cpp
  sp/parser_proto.cpp
  sp/parser_reference.cpp
  sp/parser_rule.cpp
  sp/parser_try.cpp
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
  classerror.cpp
  classfunction.cpp
  classint.cpp
  classgeneric.cpp
  classnil.cpp
  classmethod.cpp
  classnumeric.cpp
  classreference.cpp
  classrange.cpp
  classstring.cpp
  classuser.cpp
  closedsymbol.cpp
  codeerror.cpp
  collector.cpp
  collector_history.cpp
  datareader.cpp
  datawriter.cpp
  directory.cpp
  dynloader.cpp
  dynsymbol.cpp
  encodingerror.cpp
  engine.cpp
  error.cpp
  exprarray.cpp
  exprcall.cpp
  exprcompare.cpp
  exprdict.cpp
  exprdot.cpp
  exprincdec.cpp
  exprindex.cpp
  exprmultiunpack.cpp
  exprunpack.cpp
  extfunc.cpp
  errorclasses.cpp
  expression.cpp
  exprfactory.cpp
  exprmath.cpp
  exprproto.cpp
  exprref.cpp
  exprsym.cpp
  exprvalue.cpp
  famloader.cpp
  filestat.cpp
  flexyclass.cpp
  flexydict.cpp
  function.cpp
  fassert.cpp
  falconclass.cpp
  falconinstance.cpp
  falconstate.cpp
  fatal.cpp
  fstream.cpp
  genericerror.cpp
  globalsymbol.cpp
  hyperclass.cpp
  inheritance.cpp
  intcompiler.cpp
  item.cpp
  item_util.cpp
  itemarray.cpp
  itemdict.cpp
  interruptederror.cpp
  ioerror.cpp
  linkerror.cpp
  localsymbol.cpp
  metaclass.cpp
  method.cpp
  modcompiler.cpp
  modgroup.cpp
  modgroupbase.cpp
  modmap.cpp
  modloader.cpp
  modspace.cpp
  module.cpp
  multiclass.cpp
  operanderror.cpp
  overridableclass.cpp
  paramerror.cpp
  prototypeclass.cpp
  pseudofunc.cpp
  pstep.cpp
  path.cpp
  pcode.cpp
  property.cpp
  rampmode.cpp
  reader.cpp
  requirement.cpp
  rulesyntree.cpp
  sourceref.cpp
  statement.cpp
  stderrors.cpp
  stdsteps.cpp
  stmtautoexpr.cpp
  stmtfastprint.cpp
  stmtfor.cpp
  stmt_init.cpp
  stmtraise.cpp
  stmtrule.cpp
  stmtselect.cpp
  stmttry.cpp
  string.cpp
  stringstream.cpp
  stream.cpp
  sun_strtod.cpp
  symbol.cpp
  symbolmap.cpp
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
  usercarrier.cpp
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
  DESTINATION ${FALCON_INSTALL_DESTINATIONS}
)
