####################################################################
# The Falcon Programming language
#
# CMake configuration file for module sqlite3
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


if( WIN32 OR APPLE )
   set(_sqlite_default ON)
else( WIN32 OR APPLE )
   set(_sqlite_default OFF)
endif( WIN32 OR APPLE)
option(FALCON_DBI_WITH_INTERNAL_SQLITE "Use internal sqlite sources" ${_sqlite_default})


falcon_define_module( FALCON_MODULE sqlite3 )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
)


set(SRC_FILES
   sqlite3_fm.cpp
   sqlite3_mod.cpp
   sqlite3_ext.cpp
)

set(HDR_FILES
   #Headers for IDE 
   sqlite3_ext.h  
   sqlite3_mod.h  
   sqlite3_fm.h
   version.h
)

if(FALCON_DBI_WITH_INTERNAL_SQLITE)
   list(APPEND SRC_FILES sqlite3.c)
   list(APPEND HDR_FILES sqlite3.h)

   set(sqlite_definitions
      SQLITE_THREADSAFE=1
      SQLITE_ENABLE_FTS3 
      SQLITE_SECURE_DELETE
      SQLITE_ENABLE_MEMORY_MANAGEMENT
     # SQLITE_ENABLE_COLUMN_METADATA # adds API that is not wrapped right now
   )
   set_source_files_properties( sqlite.c PROPERTIES 
      COMPILE_DEFINITIONS "${sqlite_definitions}"
   )
else()
   find_package(Sqlite REQUIRED)
   include_directories(${Sqlite_INCLUDE_DIRS})
endif() 

# Target
add_library( ${FALCON_MODULE} MODULE
   ${dbi_common_files}
   ${SRC_FILES}
   ${HDR_FILES}
)

#Link
target_link_libraries( ${FALCON_MODULE}
   falcon_engine
   ${Sqlite_LIBRARIES}
)
if(NOT WIN32)
   #Non-win32 builds require pthreads
   target_link_libraries( ${FALCON_MODULE} pthread)
endif()

falcon_install_module2( ${FALCON_MODULE} dbi )

set_target_properties( ${FALCON_MODULE}
	  PROPERTIES 
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Falcon_MOD_DIR}/dbi" )
