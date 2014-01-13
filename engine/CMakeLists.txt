##################################################
# Falcon Programming Language
#
##################################################

#Engine specific define
add_definitions(-DFALCON_ENGINE_EXPORTS)

# We need config.h
include_directories( ${CMAKE_CURRENT_BINARY_DIR}/../include re2/)

message(STATUS "Compiling Falcon core on ${CMAKE_SYSTEM_NAME}" )

if(UNIX OR APPLE)
  set(SYS_SPECIFIC
    fstream_posix.cpp
    filedatampx_posix.cpp
    mt_posix.cpp
    pipe_posix.cpp
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
    filedatampx_win.cpp
    mt_win.cpp
    pipe_win.cpp
    stdstreams_win.cpp
    sys_win.cpp
    timestamp_win.cpp
    vfs_file_win.cpp
  )
endif()

#################

   
set( re2_HDRS
   re2/util/arena.h
   re2/util/atomicops.h
   re2/util/benchmark.h
   re2/util/flags.h
   re2/util/logging.h
   re2/util/mutex.h
   re2/util/pcre.h
   re2/util/random.h
   re2/util/sparse_array.h
   re2/util/sparse_set.h
   re2/util/test.h
   re2/util/utf.h
   re2/util/util.h
   re2/util/valgrind.h
   re2/re2/filtered_re2.h
   re2/re2/prefilter.h
   re2/re2/prefilter_tree.h
   re2/re2/prog.h
   re2/re2/re2.h
   re2/re2/regexp.h
   re2/re2/set.h
   re2/re2/stringpiece.h
   re2/re2/unicode_casefold.h
   re2/re2/unicode_groups.h
   re2/re2/variadic_function.h
   re2/re2/walker-inl.h
   )

set( re2_SRCS
      ${re2_HDRS}
   re2/util/arena.cc
   re2/util/hash.cc
   re2/util/rune.cc
   re2/util/stringpiece.cc
   re2/util/stringprintf.cc
   re2/util/strutil.cc
   re2/util/valgrind.cc
   re2/re2/bitstate.cc
   re2/re2/compile.cc
   re2/re2/dfa.cc
   re2/re2/filtered_re2.cc
   re2/re2/mimics_pcre.cc
   re2/re2/nfa.cc
   re2/re2/onepass.cc
   re2/re2/parse.cc
   re2/re2/perl_groups.cc
   re2/re2/prefilter.cc
   re2/re2/prefilter_tree.cc
   re2/re2/prog.cc
   re2/re2/re2.cc
   re2/re2/regexp.cc
   re2/re2/set.cc
   re2/re2/simplify.cc
   re2/re2/tostring.cc
   re2/re2/unicode_casefold.cc
   re2/re2/unicode_groups.cc
   )

#TODO re2 switch   
   
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
  builtin/dynprop.cpp
  builtin/foreach.cpp
  builtin/len.cpp
  builtin/minmax.cpp
  builtin/render.cpp
  builtin/tostring.cpp
  builtin/typeid.cpp
 
 
  cm/barrier.cpp
  cm/compile.cpp
  cm/datareader.cpp
  cm/datawriter.cpp
  cm/debugger.cpp
  cm/event.cpp
  cm/fence.cpp
  cm/gc.cpp
  cm/generator.cpp
  cm/iff.cpp
  cm/include.cpp
  cm/inspect.cpp
  cm/iterator.cpp
  cm/log.cpp
  cm/mutex.cpp
  cm/parallel.cpp
  cm/path.cpp
  cm/print.cpp
  cm/semaphore.cpp
  cm/stdfunctions.cpp
  cm/selector.cpp
  cm/syncqueue.cpp
  cm/textreader.cpp
  cm/textstream.cpp
  cm/textwriter.cpp
  cm/uri.cpp
  cm/vmcontext.cpp
  cm/vmcontextbase.cpp
  cm/vmprocess.cpp
  cm/waiter.cpp
  cm/coremodule.cpp

  sp/parser_accumulator.cpp
  sp/parser_arraydecl.cpp
  sp/parser_attribute.cpp
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
  sp/parser_global.cpp
  sp/parser_if.cpp
  sp/parser_import.cpp
  sp/parser_index.cpp
  sp/parser_list.cpp
  sp/parser_load.cpp
  sp/parser_loop.cpp
  sp/parser_namespace.cpp
  sp/parser_proto.cpp
  sp/parser_rule.cpp
  sp/parser_summon.cpp
  sp/parser_switch.cpp
  sp/parser_ternaryif.cpp
  sp/parser_try.cpp
  sp/parser_while.cpp
  sp/parsercontext.cpp
  sp/sourcelexer.cpp
  sp/sourceparser.cpp

  psteps/breakpoint.cpp
  psteps/expraccumulator.cpp
  psteps/exprarray.cpp
  psteps/exprassign.cpp
  psteps/exprautoclone.cpp
  psteps/exprbitwise.cpp
  psteps/exprcall.cpp
  psteps/exprcase.cpp
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
  psteps/expriif.cpp
  psteps/exprin.cpp
  psteps/exprincdec.cpp
  psteps/exprindex.cpp
  psteps/exprinit.cpp
  psteps/exprinherit.cpp
  psteps/exprinvoke.cpp 
  psteps/expristring.cpp
  psteps/exprlit.cpp
  psteps/exprlogic.cpp
  psteps/exprmath.cpp
  psteps/exprmultiunpack.cpp
  psteps/exprnamed.cpp
  psteps/exprneg.cpp
  psteps/exprnotin.cpp
  psteps/exproob.cpp
  psteps/exprparentship.cpp
  psteps/exprprovides.cpp
  psteps/exprrange.cpp
  psteps/exprrule.cpp
  psteps/exprproto.cpp
  psteps/exprpseudocall.cpp
  psteps/exprstripol.cpp  
  psteps/exprself.cpp
  psteps/exprstarindex.cpp
  psteps/exprsummon.cpp
  psteps/exprsym.cpp
  psteps/exprtree.cpp
  psteps/exprunpack.cpp
  psteps/exprunquote.cpp
  psteps/exprvalue.cpp

  psteps/exprvector.cpp

  psteps/stmtbreak.cpp
  psteps/stmtcontinue.cpp
  psteps/stmtfastprint.cpp
  psteps/stmtfastprintnl.cpp
  psteps/stmtfor.cpp
  psteps/stmtglobal.cpp
  psteps/stmtif.cpp
  psteps/stmtloop.cpp
  psteps/stmtraise.cpp
  psteps/stmtreturn.cpp
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
  classes/classformat.cpp
  classes/classfunction.cpp
  classes/classint.cpp
  classes/classgeneric.cpp
  classes/classmantra.cpp
  classes/classmethod.cpp
  classes/classmessagequeue.cpp
  classes/classmodule.cpp
  classes/classmodspace.cpp
  classes/classmulti.cpp
  classes/classnumber.cpp
  classes/classnumeric.cpp
  classes/classnil.cpp
  classes/classpseudodict.cpp
  classes/classrange.cpp
  classes/classrawmem.cpp
  classes/classre.cpp
  classes/classrestorer.cpp
  classes/classshared.cpp
  classes/classstorer.cpp
  classes/classstream.cpp
  classes/classstring.cpp
  classes/classstringstream.cpp  
  classes/classsymbol.cpp
  classes/classsynfunc.cpp
  classes/classtreestep.cpp
  classes/metaclass.cpp
  classes/metafalconclass.cpp
  classes/metahyperclass.cpp
  
  classes/classexpression.cpp
  classes/classstatement.cpp
  classes/classsyntree.cpp
  classes/classsymbol.cpp

  tc/transcoderc.cpp
  tc/transcoderf16.cpp
  tc/transcoderf32.cpp
  tc/transcoderutf8.cpp

  application.cpp
  attribute_helper.cpp
  attribute.cpp
  attributemap.cpp
  autocstring.cpp
  autowstring.cpp
  bom.cpp
  class.cpp
  closeddata.cpp
  closure.cpp
  collector.cpp  
  collector_history.cpp
  collectoralgorithm.cpp
  concurrencyguard.cpp
  contextgroup.cpp
  contextmanager.cpp
  datareader.cpp
  datawriter.cpp
  debugger.cpp
  delegatemap.cpp
  directory.cpp
  diskmpxfactory.cpp
  dyncompiler.cpp
  dynloader.cpp
  engine.cpp
  error.cpp
  eventmarshal.cpp
  extfunc.cpp
  expression.cpp
  fassert.cpp
  falconclass.cpp
  falconinstance.cpp
  falconinstance_private.h
  falconstate.cpp
  famloader.cpp
  fatal.cpp
  filedatampx.cpp
  filestat.cpp
  flexyclass.cpp
  flexydict.cpp
  format.cpp
  function.cpp
  fstream.cpp
  globalsmap.cpp
  hyperclass.cpp
  importdef.cpp
  instancelock.cpp
  intcompiler.cpp
  item.cpp
  item_util.cpp
  itemarray.cpp
  itemdict.cpp
  itemstack.cpp
  loaderprocess.cpp
  log.cpp
  mantra.cpp
  messagequeue.cpp
  modcompiler.cpp
  modloader.cpp
  modrequest.cpp
  modspace.cpp
  module.cpp
  multiplex.cpp
  overridableclass.cpp
  pdata.cpp
  process.cpp
  processor.cpp
  prototypeclass.cpp
  path.cpp
  pipe.cpp
  pool.cpp
  pseudofunc.cpp
  pstep.cpp
  reader.cpp
  restorer.cpp
  rootsyntree.cpp
  rulesyntree.cpp
  scheduler.cpp
  selector.cpp
  shared.cpp
  sourceref.cpp
  statement.cpp
  stdhandlers.cpp
  stdmpxfactories.cpp
  stdsteps.cpp
  storer.cpp
  stream.cpp
  streambuffer.cpp
  string.cpp
  stringstream.cpp
  stripoldata.cpp
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
  symbolmap.cpp
  vfsiface.cpp
  vfsprovider.cpp
  vm.cpp
  vmcontext.cpp
  writer.cpp
  wvmcontext.cpp
  
  ${re2_SRCS}
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
