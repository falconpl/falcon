##################################################
# Falcon Programming Language
#
##################################################

#Engine specific define
add_definitions(-DFALCON_ENGINE_EXPORTS)

# We need config.h
include_directories( ${CMAKE_CURRENT_BINARY_DIR}/../include )

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
  
  builtin/baseclass.cpp
  builtin/classname.cpp
  builtin/clone.cpp
  builtin/compare.cpp
  builtin/derivedfrom.cpp
  builtin/describe.cpp
  builtin/tostring.cpp
  builtin/typeid.cpp
  builtin/len.cpp
  builtin/minmax.cpp
 
  cm/compile.cpp
  cm/datareader.cpp
  cm/datawriter.cpp
  cm/iff.cpp
  cm/inspect.cpp
  cm/parallel.cpp
  cm/path.cpp
  cm/print.cpp
  cm/stdfunctions.cpp
  cm/textreader.cpp
  cm/textstream.cpp
  cm/textwriter.cpp
  cm/uri.cpp
  cm/coremodule.cpp

  sp/parser_arraydecl.cpp
  sp/parser_assign.cpp
  sp/parser_atom.cpp
  sp/parser_autoexpr.cpp
  sp/parser_call.cpp
  sp/parser_class.cpp
  sp/parser_deletor.cpp
  sp/parser_dynsym.cpp
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
  sp/parser_switch.cpp
  sp/parser_ternaryif.cpp
  sp/parser_try.cpp
  sp/parser_while.cpp
  sp/parsercontext.cpp
  sp/sourcelexer.cpp
  sp/sourceparser.cpp

  psteps/breakpoint.cpp
  psteps/exprarray.cpp
  psteps/exprassign.cpp
  psteps/exprbitwise.cpp
  psteps/exprcall.cpp
  psteps/exprclosure.cpp
  psteps/exprcompare.cpp
  psteps/exprcompose.cpp
  psteps/exprdict.cpp
  psteps/exprdot.cpp
  psteps/expreeq.cpp
  psteps/exprep.cpp
  psteps/exprevalret.cpp
  psteps/exprevalretexec.cpp
  psteps/exprfself.cpp
  psteps/exprfuncpower.cpp
  psteps/expriif.cpp
  psteps/exprin.cpp
  psteps/exprincdec.cpp
  psteps/exprindex.cpp
  psteps/exprinherit.cpp
  psteps/exprinvoke.cpp
  psteps/exprlit.cpp
  psteps/exprlogic.cpp
  psteps/exprmath.cpp
  psteps/exprmultiunpack.cpp
  psteps/exprneg.cpp
  psteps/exprnotin.cpp
  psteps/exproob.cpp
  psteps/exprparentship.cpp
  psteps/exprrange.cpp
  psteps/exprref.cpp
  psteps/exprproto.cpp
  psteps/exprpseudocall.cpp
  psteps/exprref.cpp
  psteps/exprself.cpp
  psteps/exprstarindex.cpp
  psteps/exprsym.cpp
  psteps/exprtree.cpp
  psteps/exprunpack.cpp
  psteps/exprunquote.cpp
  psteps/exprvalue.cpp

  psteps/exprvector.cpp

  psteps/stmtautoexpr.cpp
  psteps/stmtbreak.cpp
  psteps/stmtcontinue.cpp
  psteps/stmtfastprint.cpp
  psteps/stmtfor.cpp
  psteps/stmtif.cpp
  psteps/stmtraise.cpp
  psteps/stmtreturn.cpp
  psteps/stmtrule.cpp
  psteps/stmtselect.cpp
  psteps/stmtswitch.cpp
  psteps/stmttry.cpp
  psteps/stmtwhile.cpp
  psteps/switchlike.cpp
  
  psteps/pstep_compile.cpp

  classes/classarray.cpp
  classes/classbool.cpp
  classes/classcloseddata.cpp
  classes/classclosure.cpp
  classes/classcomposition.cpp
  classes/classdict.cpp
  classes/classerror.cpp
  classes/classfunction.cpp
  classes/classint.cpp
  classes/classgeneric.cpp
  classes/classnil.cpp
  classes/classmantra.cpp
  classes/classmethod.cpp
  classes/classmodule.cpp
  classes/classmulti.cpp
  classes/classnumeric.cpp
  classes/classrange.cpp
  classes/classrawmem.cpp
  classes/classrestorer.cpp
  classes/classshared.cpp
  classes/classstorer.cpp
  classes/classstream.cpp
  classes/classstring.cpp
  classes/classsymbol.cpp
  classes/classsynfunc.cpp
  classes/classtreestep.cpp
  classes/classrequirement.cpp
  classes/classuser.cpp
  classes/metaclass.cpp
  classes/metafalconclass.cpp
  classes/metahyperclass.cpp
  
  classes/classexpression.cpp
  classes/classstatement.cpp
  classes/classsyntree.cpp
  classes/classsymbol.cpp

  errors/accesserror.cpp
  errors/accesstypeerror.cpp
  errors/codeerror.cpp
  errors/encodingerror.cpp
  errors/errorclasses.cpp
  errors/genericerror.cpp
  errors/interruptederror.cpp
  errors/ioerror.cpp
  errors/linkerror.cpp
  errors/matherror.cpp
  errors/operanderror.cpp
  errors/paramerror.cpp
  errors/syntaxerror.cpp
  errors/unserializableerror.cpp
  errors/unsupportederror.cpp

  tc/transcoderc.cpp
  tc/transcoderutf8.cpp

  application.cpp
  autocstring.cpp
  autowstring.cpp
  bom.cpp
  class.cpp
  closeddata.cpp
  closure.cpp
  collector.cpp  
  collector_history.cpp
  collectoralgorithm.cpp
  contextgroup.cpp
  contextmanager.cpp
  datareader.cpp
  datawriter.cpp
  derivedfrom.cpp
  directory.cpp
  dynloader.cpp
  engine.cpp
  error.cpp
  extfunc.cpp
  expression.cpp
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
  hyperclass.cpp
  importdef.cpp
  intcompiler.cpp
  item.cpp
  item_util.cpp
  itemarray.cpp
  itemdict.cpp
  log.cpp
  mantra.cpp
  method.cpp
  modcompiler.cpp
  modloader.cpp
  modrequest.cpp
  modspace.cpp
  module.cpp
  overridableclass.cpp
  process.cpp
  processor.cpp
  prototypeclass.cpp
  path.cpp
  pool.cpp
  property.cpp
  pseudofunc.cpp
  pstep.cpp
  reader.cpp
  restorer.cpp
  requirement.cpp
  rootsyntree.cpp
  rulesyntree.cpp
  shared.cpp
  sourceref.cpp
  statement.cpp
  stderrors.cpp
  stdsteps.cpp
  storer.cpp
  string.cpp
  stringstream.cpp
  stream.cpp
  streambuffer.cpp
  sun_strtod.cpp
  symbol.cpp
  synclasses.cpp
  synfunc.cpp
  syntree.cpp
  textreader.cpp
  textwriter.cpp
  timestamp.cpp  
  transcoder.cpp
  trace.cpp
  tracestep.cpp
  treestep.cpp
  uri.cpp
  usercarrier.cpp
  vardatamap.cpp
  variable.cpp
  varmap.cpp
  vfsiface.cpp
  vfsprovider.cpp
  vm.cpp
  vmcontext.cpp
  vmtimer.cpp
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
