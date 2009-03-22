##################################################
# Falcon Programming Language
#
# Faltest
##################################################

#Engine specific define
ADD_DEFINITIONS(-DFALCON_ENGINE_EXPORTS)


#Specific system files
IF(UNIX OR APPLE)
   SET( SYS_SPECIFIC
         baton_posix.cpp
         dir_sys_unix.cpp
         fstream_sys_unix.cpp
         mt_posix.cpp
         stdstreams_unix.cpp
         time_sys_unix.cpp
         vm_sys_posix.cpp
         vfs_file_unix.cpp
    )

   #for solaris system flavor, add sys_solaris.cpp
   IF( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         sys_solaris.cpp
      )
   ELSE( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         sys_unix.cpp
      )
   ENDIF( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )

   #DL is different for unix and mac
   IF(APPLE)
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         heap_bsd.cpp
         dll_mac.cpp )
   ELSE(APPLE)
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         heap_posix.cpp
         dll_dl.cpp )
   ENDIF(APPLE)

   #We need bison.
   IF(NOT BISON_EXECUTABLE)
      FIND_PROGRAM(BISON_EXECUTABLE bison)
      IF (NOT BISON_EXECUTABLE)
         MESSAGE(FATAL_ERROR "BISON not found - aborting")
      ENDIF (NOT BISON_EXECUTABLE)
   ENDIF(NOT BISON_EXECUTABLE)

   # Adding bison parsers.
   ADD_CUSTOM_TARGET(SrcParser echo "Creating src_parser.cpp")
   ADD_CUSTOM_TARGET(SrcParser_h echo "Creating src_parser.h")

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

   ADD_CUSTOM_COMMAND(
      SOURCE ${FALCON_SRC_TREE}/core/engine/src_parser.hpp
      COMMAND cp
      ARGS ${FALCON_SRC_TREE}/core/engine/src_parser.hpp ${FALCON_SRC_TREE}/core/engine/src_parser.h
      TARGET SrcParser_h
      OUTPUTS
         ${FALCON_SRC_TREE}/core/engine/src_parser.h )


ELSEIF(WIN32)

   SET( SYS_SPECIFIC
      baton_win.cpp
      dir_sys_win.cpp
      dll_win.cpp
      fstream_sys_win.cpp
      heap_win.cpp
      mt_win.cpp
      stdstreams_win.cpp
      sys_win.cpp
      time_sys_win.cpp
      vm_sys_win.cpp
      vfs_file_win.cpp
      falcon_engine.rc
      )

ENDIF(UNIX OR APPLE)

INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(core_module)

SET( CORE_MODULE_SOURCES
   core_module/array_ext.cpp
   core_module/cmdlineparser.cpp
   core_module/core_module.cpp
   core_module/coroutine_ext.cpp
   core_module/dict_ext.cpp
   core_module/dir_ext.cpp
   core_module/envvars.cpp
   core_module/error_ext.cpp
   core_module/fal_include.cpp
   core_module/file_ext.cpp
   core_module/format_ext.cpp
   core_module/function_ext.cpp
   core_module/functional_ext.cpp
   core_module/gc_ext.cpp
   core_module/input.cpp
   core_module/inspect.cpp
   core_module/item_ext.cpp
   core_module/iterator_ext.cpp
   core_module/list.cpp
   core_module/math.cpp
   core_module/membuf_ext.cpp
   core_module/message_ext.cpp
   core_module/oob_ext.cpp
   core_module/pagedict_ext.cpp
   core_module/param_ext.cpp
   core_module/path_ext.cpp
   core_module/print.cpp
   core_module/random.cpp
   core_module/seconds.cpp
   core_module/serialize.cpp
   core_module/string_ext.cpp
   core_module/stringstream_ext.cpp
   core_module/sys_ext.cpp
   core_module/table.cpp
   core_module/time_ext.cpp
   core_module/transcode_ext.cpp
   core_module/tokenizer_ext.cpp
   core_module/uri_ext.cpp
   core_module/vminfo_ext.cpp
)

ADD_LIBRARY( falcon_engine SHARED
   src_parser.cpp
   autocstring.cpp
   autowstring.cpp
   basealloc.cpp
   cacheobject.cpp
   cdict.cpp
   crobject.cpp
   compiler.cpp
   corearray.cpp
   coreclass.cpp
   corefunc.cpp
   coreobject.cpp
   coreslot.cpp
   coretable.cpp
   deptab.cpp
   error.cpp
   falcondata.cpp
   falconobject.cpp
   fassert.cpp
   filestat.cpp
   format.cpp
   garbageable.cpp
   gcalloc.cpp
   gencode.cpp
   genericlist.cpp
   genericmap.cpp
   genericvector.cpp
   genhasm.cpp
   gentree.cpp
   globals.cpp
   intcomp.cpp
   item.cpp
   item_co.cpp
   itemlist.cpp
   itemserial.cpp
   itemtraits.cpp
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
   rampmode.cpp
   reflectobject.cpp
   rosstream.cpp
   runtime.cpp
   service.cpp
   smba.cpp
   src_lexer.cpp
   stream.cpp
   streambuffer.cpp
   string.cpp
   stringitem.cpp
   stringstream.cpp
   strtable.cpp
   symbol.cpp
   symtab.cpp
   syntree.cpp
   timestamp.cpp
   traits.cpp
   transcoding.cpp
   tokenizer.cpp
   uri.cpp
   vfsprovider.cpp
   vmcontext.cpp
   vm.cpp
   vmmaps.cpp
   vmmsg.cpp
   vm_run.cpp
   vmsema.cpp

   ${CORE_MODULE_SOURCES}

   ${SYS_SPECIFIC}
   )

IF(BISON_EXECUTABLE)
   ADD_DEPENDENCIES( falcon_engine SrcParser SrcParser_h )
ENDIF(BISON_EXECUTABLE)

SET_TARGET_PROPERTIES(falcon_engine
   PROPERTIES
      VERSION "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}"
      SOVERSION "${FALCON_SONAME_VERSION}" )

#In unix and mac we have to add extra libraries
IF(UNIX OR APPLE)
   TARGET_LINK_LIBRARIES(falcon_engine m)

   #also dl for linux and solaris
   # and RT only for solaris and non-bsd systems
   IF(APPLE)
      SET(SYS_LIBS
        dl
        pthread )
   ELSE(APPLE)
      SET(SYS_LIBS
         dl
         pthread
         rt)
   ENDIF(APPLE)

   TARGET_LINK_LIBRARIES(falcon_engine ${SYS_LIBS})

      #Finally, link everything to the lib dir
   INSTALL( TARGETS falcon_engine
         DESTINATION "${FALCON_LIB_DIR}" )

ELSE(UNIX OR APPLE)
   INSTALL( TARGETS falcon_engine
         DESTINATION "${FALCON_MOD_INSTALL}" )
ENDIF(UNIX OR APPLE)
