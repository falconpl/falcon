####################################################################
# The Falcon Programming language
#
# CMake configuration file for module mysql
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


falcon_define_module( FALCON_MODULE fbsql )

set( firebird_include_search_path
      ${FIREBIRD_INCLUDE_DIR}
      /usr/local/include
      /usr/include )
      
# Find FB includes
FIND_PATH(
   FIREBIRD_INC_DIR ibase.h
      ${firebird_include_search_path}
  )

  
if (NOT FIREBIRD_INC_DIR)
   MESSAGE(FATAL_ERROR "Can't find ibase.h header file in ${firebird_include_search_path}" )
endif()

# Find MySQL Client Library
SET( firebird_lib_search_path  "${FIREBIRD_LIB_DIR}" /usr/local/lib /usr/lib  )

FIND_LIBRARY( FIREBIRD_LIBRARY
  NAMES libfbclient.so
  PATHS ${firebird_lib_search_path}
)

if (NOT FIREBIRD_LIBRARY)
   MESSAGE(FATAL_ERROR "Can't find library libfbclient.so in ${firebird_lib_search_path}" )
endif()


MESSAGE(STATUS "Found ibase.h in ${FIREBIRD_INC_DIR}")
MESSAGE(STATUS "Found firebird client library ${FIREBIRD_LIBRARY}")

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(../dbi)
INCLUDE_DIRECTORIES(${FIREBIRD_INC_DIR})


# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE

   ${dbi_common_files}

   fbsql.cpp
   fbsql_ext.cpp
   fbsql_mod.cpp
)

# Link
IF(MSVC)
  SET(FIREBIRD_OPT_LIBS wsock32)
ENDIF(MSVC)

TARGET_LINK_LIBRARIES( ${FALCON_MODULE}  falcon_engine ${FIREBIRD_OPT_LIBS} ${FIREBIRD_LIBRARY})
falcon_install_module2( ${FALCON_MODULE} dbi )
set_target_properties( ${FALCON_MODULE}
	  PROPERTIES 
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Falcon_MOD_DIR}/dbi" )
      
