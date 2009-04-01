####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ODBC
####################################################################

FALCON_DEFINE_MODULE( MODULE odbc )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Find ODBC
SET(ODBC_INCLUDE_SEARCH_PATH
  ${ODBC_INCLUDE_DIR}
  "$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/include"
)
FIND_PATH(ODBC_REAL_INCLUDE_DIR odbcss.h ${ODBC_INCLUDE_SEARCH_PATH})

# Find ODBC Client Library
SET(ODBC_LIB_SEARCH_PATH
   ${ODBC_LIB_DIR}
   "$ENV{ProgramFiles}/Microsoft Visual Studio 8/VC/PlatformSDK/lib"
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

INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

# Target
ADD_LIBRARY( ${MODULE} MODULE
   odbc.cpp
   odbc_ext.cpp
   odbc_srv.cpp
)

# Link
IF(MSVC)
  SET_TARGET_PROPERTIES(${MODULE} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:library")
ENDIF(MSVC)

TARGET_LINK_LIBRARIES( ${MODULE} falcon_engine ${ODBC_OPT_LIBS} ${ODBC_LIBRARY} ${ODBCCP_LIBRARY})
FALCON_INSTALL_MODULE( ${MODULE} )
