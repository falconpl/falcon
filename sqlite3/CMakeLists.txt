####################################################################
# The Falcon Programming language
#
# CMake configuration file for module sqlite3
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

falcon_define_module( FALCON_MODULE sqlite3 )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)

# Target
add_library( ${FALCON_MODULE} MODULE
   ${dbi_common_files}

   sqlite3_fm.cpp
   sqlite3_mod.cpp
   sqlite3_ext.cpp
   sqlite3_srv.cpp
   sqlite3.c
   
   #Headers for IDE 
   sqlite3_ext.h  
   sqlite3_mod.h  
   version.h
   sqlite3.h

)
#
set(sqlite_definitions
  SQLITE_THREADSAFE=1
  SQLITE_ENABLE_FTS3 
  SQLITE_SECURE_DELETE
  SQLITE_ENABLE_MEMORY_MANAGEMENT
  # SQLITE_ENABLE_COLUMN_METADATA # adds API that is not wrapped right now
)
set_source_files_properties( sqlite.c 
  PROPERTIES 
    COMPILE_DEFINITIONS "${sqlite_definitions}"
)

#Link
if(WIN32)
   target_link_libraries(${FALCON_MODULE} falcon_engine)
else()
   #Non-win32 builds require pthreads
   target_link_libraries(${FALCON_MODULE} falcon_engine pthread)
endif()

falcon_install_module( ${FALCON_MODULE} )
