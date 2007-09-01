PROJECT(falcon_engine)

INCLUDE_DIRECTORIES(BEFORE $ENV{FALCON_BUILD_TREE}/include)
INCLUDE_DIRECTORIES(BEFORE ../include)

#Engine specific define
ADD_DEFINITIONS(-DFALCON_ENGINE_EXPORTS)

#Specific system files
IF(UNIX OR MAC)
   SET( SYS_SPECIFIC
         dir_sys_unix.cpp
         fstream_sys_unix.cpp
         stdstreams_unix.cpp
         sys_unix.cpp
         time_sys_unix.cpp
    )

   #DL is different for unix and mac
   IF(UNIX)
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         dll_dl.cpp )
   ELSE(UNIX)
      SET( SYS_SPECIFIC
         ${SYS_SPECIFIC}
         dll_win.cpp )
   ENDIF(UNIX)

   # Make in source tree
   # We're "hard linking" this as pregenerated, eventually re-generating them when not there.
   ADD_CUSTOM_COMMAND( OUTPUT $ENV{FALCON_SRC_TREE}/core/engine/fasm_parser.cpp
      COMMAND bison $ENV{FALCON_SRC_TREE}/core/engine/fasm_parser.yy -o $ENV{FALCON_SRC_TREE}/core/engine/fasm_parser.cpp
      COMMAND mv -f $ENV{FALCON_SRC_TREE}/core/engine/fasm_parser.hpp $ENV{FALCON_SRC_TREE}/core/engine/fasm_parser.h )

   ADD_CUSTOM_COMMAND( OUTPUT $ENV{FALCON_SRC_TREE}/core/engine/src_parser.cpp
      COMMAND bison $ENV{FALCON_SRC_TREE}/core/engine/src_parser.yy -o $ENV{FALCON_SRC_TREE}/core/engine/src_parser.cpp
      COMMAND mv -f $ENV{FALCON_SRC_TREE}/core/engine/src_parser.hpp $ENV{FALCON_SRC_TREE}/core/engine/src_parser.h )

ELSEIF(WIN32)

   #todo: add prebuilt
   SET( SYS_SPECIFIC
      dir_sys_win.cpp
      dll_win.cpp
      fstream_sys_win.cpp
      stdstreams_win.cpp
      sys_win.cpp
      time_sys_win.cpp
      falcon_engine.rc
      )

ENDIF(UNIX OR MAC)

INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES($ENV{FALCON_BUILD_TREE}/core/engine/)


ADD_LIBRARY( falcon_engine SHARED
   fasm_parser.cpp
   src_parser.cpp

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
   falcon_engine.rc
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
   itemserial.cpp
   itemtraits.cpp
   labeldef.cpp
   lineardict.cpp
   linemap.cpp
   ltree.cpp
   memhash.cpp
   memory.cpp
   mempool.cpp
   modloader.cpp
   module.cpp
   pagedict.cpp
   proptable.cpp
   pseudo.cpp
   rosstream.cpp
   runtime.cpp
   src_lexer.cpp
   stackframe.cpp
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
   userdata.cpp
   vmcontext.cpp
   vm.cpp
   vmmaps.cpp
   vm_run.cpp
   vmsema.cpp

   ${SYS_SPECIFIC}
   )

SET_TARGET_PROPERTIES(falcon_engine
   PROPERTIES
      VERSION "${FALCON_SONAME_VERSION}"
      SOVERSION "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}" )

SET(CMAKE_INSTALL_PREFIX "")
FALCON_LIB_INSTALL( falcon_engine )

#In unix and mac we have to add extra libraries
IF(UNIX OR MAC)
   TARGET_LINK_LIBRARIES(falcon_engine m)
   #also dl for linux
   IF(UNIX)
      TARGET_LINK_LIBRARIES(falcon_engine dl)
   ENDIF(UNIX)

   #Finally, link everything to the lib dir

ENDIF(UNIX OR MAC)
