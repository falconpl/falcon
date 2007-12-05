PROJECT(falcon_engine)

#Engine specific define
ADD_DEFINITIONS(-DFALCON_ENGINE_EXPORTS)
#set common include directory
FALCON_INCLUDE_SETTINGS()

# BISON a .ll file
# search BISON
MACRO(FIND_BISON)
   IF(NOT BISON_EXECUTABLE)
      FIND_PROGRAM(BISON_EXECUTABLE bison)
      IF (NOT BISON_EXECUTABLE)
         MESSAGE(FATAL_ERROR "BISON not found - aborting")
      ENDIF (NOT BISON_EXECUTABLE)
   ENDIF(NOT BISON_EXECUTABLE)
ENDMACRO(FIND_BISON)

MACRO( ADD_BISON_FILES _sources )
   FIND_BISON()
   FOREACH (_current_FILE ${ARGN})
      GET_FILENAME_COMPONENT(_in ${_current_FILE} ABSOLUTE)
      GET_FILENAME_COMPONENT(_basename ${_current_FILE} NAME_WE)
      SET(_out ${CMAKE_CURRENT_SOURCE_DIR}/${_basename}.cpp)
      SET(_out_h ${CMAKE_CURRENT_SOURCE_DIR}/${_basename}.h)
      ADD_CUSTOM_COMMAND(
         OUTPUT ${_out}
         COMMAND ${BISON_EXECUTABLE}
         ARGS -o${_out} --defines=${_out_h} ${_in}
         DEPENDS ${_in} )

      SET(${_sources} ${${_sources}} ${_out} )
   ENDFOREACH (_current_FILE)
ENDMACRO(ADD_BISON_FILES)


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

   ADD_BISON_FILES( bison_src
      fasm_parser.yy
      src_parser.yy
   )

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

   SET( bison_src
      fasm_parser.cpp
      src_parser.cpp
   )

ENDIF(UNIX OR MAC)

INCLUDE_DIRECTORIES(.)

ADD_LIBRARY( falcon_engine SHARED
   ${bison_src}

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
      VERSION "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}"
      SOVERSION "${FALCON_SONAME_VERSION}" )

#In unix and mac we have to add extra libraries
IF(UNIX OR MAC)
   TARGET_LINK_LIBRARIES(falcon_engine m)
   #also dl for linux
   IF(UNIX)
      TARGET_LINK_LIBRARIES(falcon_engine dl)
   ENDIF(UNIX)

   #Finally, link everything to the lib dir

ENDIF(UNIX OR MAC)

SET(CMAKE_INSTALL_PREFIX "")
FALCON_LIB_INSTALL( falcon_engine )
