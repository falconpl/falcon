##################################################
# Falcon Programming Language
#
# Faltest
##################################################

#Engine specific define
ADD_DEFINITIONS(-DFALCON_ENGINE_EXPORTS)


#Specific system files
IF(UNIX OR MAC)
   SET( SYS_SPECIFIC
         dir_sys_unix.cpp
         fstream_sys_unix.cpp
         stdstreams_unix.cpp
         time_sys_unix.cpp
         vm_sys_posix.cpp
    )

   #for solaris system flavor, add sys_solaris.cpp
   IF( "${FALCON_UNIX_FLAVOR}" STREQUAL "SOLARIS" )
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         sys_solaris.cpp
      )
   ELSE( "${FALCON_UNIX_FLAVOR}" STREQUAL "SOLARIS" )
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         sys_unix.cpp
      )
   ENDIF( "${FALCON_UNIX_FLAVOR}" STREQUAL "SOLARIS" )

   #DL is different for unix and mac
   IF(UNIX)
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         dll_dl.cpp )
   ELSE(UNIX)
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         dll_mac.cpp )
   ENDIF(UNIX)

   #We need bison.
   IF(NOT BISON_EXECUTABLE)
      FIND_PROGRAM(BISON_EXECUTABLE bison)
      IF (NOT BISON_EXECUTABLE)
         MESSAGE(FATAL_ERROR "BISON not found - aborting")
      ENDIF (NOT BISON_EXECUTABLE)
   ENDIF(NOT BISON_EXECUTABLE)

   # Adding bison parsers.
   ADD_CUSTOM_TARGET(SrcParser echo "Creating src_parser.cpp")
   ADD_CUSTOM_TARGET(FasmParser echo "Creating fasm_parser.cpp")
   ADD_CUSTOM_TARGET(SrcParser_h echo "Creating src_parser.h")
   ADD_CUSTOM_TARGET(FasmParser_h echo "Creating fasm_parser.h")

   #builds in-source always; we want it in svn.
   ADD_CUSTOM_COMMAND(
      SOURCE ${FALCON_SRC_TREE}/core/engine/src_parser.yy
      COMMAND ${BISON_EXECUTABLE}
      ARGS -y ${FALCON_SRC_TREE}/core/engine/src_parser.yy -o ${FALCON_SRC_TREE}/core/engine/src_parser.cpp
      TARGET SrcParser
      DEPENDS ${FALCON_SRC_TREE}/core/include/falcon/syntree.h
      OUTPUTS
         ${FALCON_SRC_TREE}/core/engine/src_parser.cpp
         ${FALCON_SRC_TREE}/core/engine/src_parser.hpp)

   #builds in-source always; we want it in svn.
   ADD_CUSTOM_COMMAND(
      SOURCE ${FALCON_SRC_TREE}/core/engine/fasm_parser.yy
      COMMAND ${BISON_EXECUTABLE}
      ARGS -y ${FALCON_SRC_TREE}/core/engine/fasm_parser.yy -o ${FALCON_SRC_TREE}/core/engine/fasm_parser.cpp
      TARGET FasmParser
      OUTPUTS
         ${FALCON_SRC_TREE}/core/engine/fasm_parser.cpp
         ${FALCON_SRC_TREE}/core/engine/fasm_parser.hpp )

   ADD_CUSTOM_COMMAND(
      SOURCE ${FALCON_SRC_TREE}/core/engine/src_parser.hpp
      COMMAND cp
      ARGS ${FALCON_SRC_TREE}/core/engine/src_parser.hpp ${FALCON_SRC_TREE}/core/engine/src_parser.h
      TARGET SrcParser_h
      OUTPUTS
         ${FALCON_SRC_TREE}/core/engine/src_parser.h )

   ADD_CUSTOM_COMMAND(
      SOURCE ${FALCON_SRC_TREE}/core/engine/fasm_parser.hpp
      COMMAND cp
      ARGS ${FALCON_SRC_TREE}/core/engine/fasm_parser.hpp ${FALCON_SRC_TREE}/core/engine/fasm_parser.h
      TARGET FasmParser_h
      OUTPUTS
         ${FALCON_SRC_TREE}/core/engine/fasm_parser.h )

ELSEIF(WIN32)

   SET( SYS_SPECIFIC
      dir_sys_win.cpp
      dll_win.cpp
      fstream_sys_win.cpp
      stdstreams_win.cpp
      sys_win.cpp
      time_sys_win.cpp
      vm_sys_win.cpp
      falcon_engine.rc
      )

ENDIF(UNIX OR MAC)

INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(core_module)

SET( CORE_MODULE_SOURCES
   core_module/array.cpp
   core_module/attribute_ext.cpp
   core_module/cmdlineparser.cpp
   core_module/core_module.cpp
   core_module/coroutine_ext.cpp
   core_module/dict.cpp
   core_module/dir_ext.cpp
   core_module/envvars.cpp
   core_module/error_ext.cpp
   core_module/fbom.cpp
   core_module/file_ext.cpp
   core_module/format_ext.cpp
   core_module/functional_ext.cpp
   core_module/gc_ext.cpp
   core_module/indirect.cpp
   core_module/input.cpp
   core_module/inspect.cpp
   core_module/item_ext.cpp
   core_module/itemcopy.cpp
   core_module/iterator_ext.cpp
   core_module/list.cpp
   core_module/math.cpp
   core_module/membuf_ext.cpp
   core_module/messages.cpp
   core_module/oob_ext.cpp
   core_module/pagedict_ext.cpp
   core_module/param_ext.cpp
   core_module/print.cpp
   core_module/random.cpp
   core_module/seconds.cpp
   core_module/serialize.cpp
   core_module/string.cpp
   core_module/stringstream_ext.cpp
   core_module/sys_ext.cpp
   core_module/time_ext.cpp
   core_module/transcode_ext.cpp
   core_module/uri_ext.cpp
   core_module/vminfo_ext.cpp
)

ADD_LIBRARY( falcon_engine SHARED
   fasm_parser.cpp
   src_parser.cpp

   attribute.cpp
   autocstring.cpp
   autowstring.cpp
   basealloc.cpp
   bufferedstream.cpp
   cobject.cpp
   compiler.cpp
   corearray.cpp
   coreclass.cpp
   deferrorhandler.cpp
   deptab.cpp
   detmempool.cpp
   enginedata.cpp
   engstrings.cpp
   error.cpp
   falcondata.cpp
   fasm_comp.cpp
   fasm_lexer.cpp
   fassert.cpp
   filestat.cpp
   flcloader.cpp
   format.cpp
   garbageable.cpp
   gencode.cpp
   genericlist.cpp
   genericmap.cpp
   genericvector.cpp
   genhasm.cpp
   gentree.cpp
   intcomp.cpp
   item.cpp
   itemlist.cpp
   itemserial.cpp
   itemtraits.cpp
   labeldef.cpp
   lineardict.cpp
   linemap.cpp
   ltree.cpp
   membuf.cpp
   memhash.cpp
   memory.cpp
   mempool.cpp
   modloader.cpp
   module.cpp
   pagedict.cpp
   path.cpp
   proptable.cpp
   pseudo.cpp
   rosstream.cpp
   runtime.cpp
   src_lexer.cpp
   stream.cpp
   string.cpp
   stringstream.cpp
   strtable.cpp
   symbol.cpp
   symtab.cpp
   syntree.cpp
   timestamp.cpp
   traits.cpp
   transcoding.cpp
   uri.cpp
   userdata.cpp
   vmcontext.cpp
   vm.cpp
   vmmaps.cpp
   vm_run.cpp
   vmsema.cpp

   ${CORE_MODULE_SOURCES}

   ${SYS_SPECIFIC}
   )

IF(BISON_EXECUTABLE)
   ADD_DEPENDENCIES( falcon_engine SrcParser FasmParser SrcParser_h FasmParser_h )
ENDIF(BISON_EXECUTABLE)

SET_TARGET_PROPERTIES(falcon_engine
   PROPERTIES
      VERSION "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}"
      SOVERSION "${FALCON_SONAME_VERSION}" )

#In unix and mac we have to add extra libraries
IF(UNIX OR MAC)
   TARGET_LINK_LIBRARIES(falcon_engine m)

   #also dl for linux and solaris
   IF(UNIX)
      # and RT only for solaris
      IF( "${FALCON_UNIX_FLAVOR}" STREQUAL "SOLARIS" )
         TARGET_LINK_LIBRARIES(falcon_engine dl rt)
      ELSE( "${FALCON_UNIX_FLAVOR}" STREQUAL "SOLARIS" )
         TARGET_LINK_LIBRARIES(falcon_engine dl)
      ENDIF( "${FALCON_UNIX_FLAVOR}" STREQUAL "SOLARIS" )
   ENDIF(UNIX)

   #Finally, link everything to the lib dir
INSTALL( TARGETS falcon_engine
         DESTINATION "${FALCON_LIB_DIR}" )

ELSE(UNIX OR MAC)
INSTALL( TARGETS falcon_engine
         DESTINATION "${FALCON_MOD_INSTALL}" )
ENDIF(UNIX OR MAC)
