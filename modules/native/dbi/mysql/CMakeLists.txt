####################################################################
# The Falcon Programming language
#
# CMake configuration file for module mysql
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

option( FALCON_DBI_MYSQL_MY_RPATH "Append mysqlclient library location to install RPATH" ON)

falcon_define_module( FALCON_MODULE mysql )

# Find MYSQL
SET(MYSQL_INCLUDE_SEARCH_PATH
  ${MYSQL_INCLUDE_DIR}
  /usr/include/mysql
  /usr/local/include/mysql
)
FIND_PATH(MYSQL_REAL_INCLUDE_DIR mysql.h ${MYSQL_INCLUDE_SEARCH_PATH})

# Find MySQL Client Library
SET(MYSQL_LIB_SEARCH_PATH ${MYSQL_LIB_DIR} /usr/lib /usr/lib/mysql /usr/local/lib /usr/local/lib/mysql)
SET(MYSQL_LIB_NAMES mysqlclient mysqlclient_r)
FIND_LIBRARY(MYSQL_LIBRARY
  NAMES ${MYSQL_LIB_NAMES}
  PATHS ${MYSQL_LIB_SEARCH_PATH}
)

IF (MYSQL_REAL_INCLUDE_DIR AND MYSQL_LIBRARY)
  INCLUDE_DIRECTORIES(BEFORE ${MYSQL_REAL_INCLUDE_DIR})
ELSE (MYSQL_REAL_INCLUDE_DIR AND MYSQL_LIBRARY)
  IF(NOT MYSQL_REAL_INCLUDE_DIR)
    MESSAGE(SEND_ERROR "MySQL include path was not found")
    MESSAGE(SEND_ERROR "Looked in: ${MYSQL_INCLUDE_SEARCH_PATH}")
  ENDIF (NOT MYSQL_REAL_INCLUDE_DIR)
  IF (NOT MYSQL_LIBRARY)
    MESSAGE(SEND_ERROR "MySQL client library was not found")
    MESSAGE(SEND_ERROR "Looked for: ${MYSQL_LIB_NAMES}")
    MESSAGE(SEND_ERROR "Looked in: ${MYSQL_LIB_SEARCH_PATH}")
  ENDIF (NOT MYSQL_LIBRARY)
  MESSAGE(FATAL_ERROR "Cannot build MySQL DBI module")
ENDIF (MYSQL_REAL_INCLUDE_DIR AND MYSQL_LIBRARY)

MESSAGE(STATUS "Found mysql.h in ${MYSQL_REAL_INCLUDE_DIR}")
MESSAGE(STATUS "Found mysql client library, ${MYSQL_LIBRARY}")

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(../dbi)

# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE

   ${dbi_common_files}

   mysql.cpp
   mysql_ext.cpp
   mysql_mod.cpp
)

# Link
IF(MSVC)
  SET(MYSQL_OPT_LIBS wsock32)
  SET_TARGET_PROPERTIES(${FALCON_MODULE}  PROPERTIES LINK_FLAGS "/NODEFAULTLIB:library")
ENDIF(MSVC)

TARGET_LINK_LIBRARIES(${FALCON_MODULE}  falcon_engine ${MYSQL_OPT_LIBS} ${MYSQL_LIBRARY})

if(FALCON_DBI_MYSQL_MY_RPATH)
   get_filename_component(rpath "${MYSQL_LIBRARY}" PATH )
   set_target_properties( 
	${FALCON_MODULE} 
        PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${rpath}" )
   message( "DBI.Mysql Setting install rpath to ${CMAKE_INSTALL_RPATH}:${rpath}" )
endif()

falcon_install_module2( ${FALCON_MODULE} dbi )

set_target_properties( ${FALCON_MODULE}
	  PROPERTIES 
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Falcon_MOD_DIR}/dbi" )
      