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

   #builds in-source always; we want it in svn.
   ADD_CUSTOM_COMMAND(
      SOURCE ${FALCON_SRC_TREE}/core/engine/src_parser.yy
      COMMAND ${BISON_EXECUTABLE}
      ARGS -y ${FALCON_SRC_TREE}/core/engine/src_parser.yy -o ${FALCON_SRC_TREE}/core/engine/src_parser.cpp
      TARGET SrcParser
      DEPENDS ${FALCON_SRC_TREE}/core/include/falcon/syntree.h
      OUTPUTS ${FALCON_SRC_TREE}/core/engine/src_parser.cpp )

   #builds in-source always; we want it in svn.
   ADD_CUSTOM_COMMAND(
      SOURCE ${FALCON_SRC_TREE}/core/engine/src_parser.yy
      COMMAND ${BISON_EXECUTABLE}
      ARGS -y ${FALCON_SRC_TREE}/core/engine/fasm_parser.yy -o ${FALCON_SRC_TREE}/core/engine/fasm_parser.cpp
      TARGET FasmParser
      OUTPUTS ${FALCON_SRC_TREE}/core/engine/fasm_parser.cpp )

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
   core_ext.cpp
   deferrorhandler.cpp
   deptab.cpp
   detmempool.cpp
   enginedata.cpp
   engstrings.cpp
   error.cpp
   fasm_comp.cpp
   fasm_lexer.cpp
   fassert.cpp
   fbom.cpp
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
   suserdata.cpp
   symbol.cpp
   symtab.cpp
   syntree.cpp
   timestamp.cpp
   traits.cpp
   transcoding.cpp
   userdata.cpp
   uri.cpp
   vmcontext.cpp
   vm.cpp
   vmmaps.cpp
   vm_run.cpp
   vmsema.cpp

   ${SYS_SPECIFIC}
   )

IF(BISON_EXECUTABLE)
   ADD_DEPENDENCIES( falcon_engine SrcParser FasmParser )
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
