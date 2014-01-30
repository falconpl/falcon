####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ODBC
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FALCON_DEFINE_MODULE( FALCON_MODULE odbc )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Find ODBC
SET(ODBC_INCLUDE_SEARCH_PATH
  ${ODBC_INCLUDE_DIR}
  "$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/include"
  "$ENV{ProgramFiles}/Microsoft Visual Studio 9/VC/PlatformSDK/include"
  "$ENV{ProgramFiles}/Microsoft SDKs/Windows/v7.0A/Include"
  "$ENV{ProgramFiles}/Microsoft Platform SDK/Include"
)
FIND_PATH(ODBC_REAL_INCLUDE_DIR odbcss.h ${ODBC_INCLUDE_SEARCH_PATH})

# Find ODBC Client Library
SET(ODBC_LIB_SEARCH_PATH
   ${ODBC_LIB_DIR}
   "$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/lib"
   "$ENV{ProgramFiles}/Microsoft Visual Studio 9/VC/PlatformSDK/lib"
   "$ENV{ProgramFiles}/Microsoft SDKs/Windows/v7.0A/Lib"
   "$ENV{ProgramFiles}/Microsoft Platform SDK/Lib"
  )

FIND_LIBRARY(ODBC_LIBRARY
  NAMES odbc32
  PATHS ${ODBC_LIB_SEARCH_PATH}
)

IF (ODBC_REAL_INCLUDE_DIR AND ODBC_LIBRARY)
  INCLUDE_DIRECTORIES(BEFORE ${ODBC_REAL_INCLUDE_DIR})
ELSE (ODBC_REAL_INCLUDE_DIR AND ODBC_LIBRARY)
  IF(NOT ODBC_REAL_INCLUDE_DIR)
    MESSAGE(SEND_ERROR "ODBC include path was not found")
    MESSAGE(SEND_ERROR "Looked in: ${ODBC_INCLUDE_SEARCH_PATH}")
  ENDIF (NOT ODBC_REAL_INCLUDE_DIR)
  IF (NOT ODBC_LIBRARY)
    MESSAGE(SEND_ERROR "ODBC client library was not found")
    MESSAGE(SEND_ERROR "Looked for: ${ODBC_LIB_NAMES}")
    MESSAGE(SEND_ERROR "Looked in: ${ODBC_LIB_SEARCH_PATH}")
  ENDIF (NOT ODBC_LIBRARY)
  MESSAGE(FATAL_ERROR "Cannot build ODBC DBI module")
ENDIF (ODBC_REAL_INCLUDE_DIR AND ODBC_LIBRARY)

#  we know we have found it
FIND_LIBRARY(ODBCCP_LIBRARY
  NAMES odbccp32
  PATHS ${ODBC_LIB_SEARCH_PATH}
)

MESSAGE(STATUS "Found ODBC.h in ${ODBC_REAL_INCLUDE_DIR}")
MESSAGE(STATUS "Found ODBC client library, ${ODBC_LIBRARY}")

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)

# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE
   ${dbi_common_files}

   odbc_fm.cpp
   odbc_ext.cpp
   odbc_mod.cpp

   odbc_ext.h
   odbc_mod.h
   odbc_fm.h
)

# Link
IF(MSVC)
  SET_TARGET_PROPERTIES(${FALCON_MODULE} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:library")
ENDIF(MSVC)

TARGET_LINK_LIBRARIES( ${FALCON_MODULE} falcon_engine ${ODBC_OPT_LIBS} ${ODBC_LIBRARY} ${ODBCCP_LIBRARY})
FALCON_INSTALL_MODULE2( ${FALCON_MODULE} dbi )

set_target_properties( ${FALCON_MODULE}
	  PROPERTIES 
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${Falcon_MOD_DIR}/dbi" )
