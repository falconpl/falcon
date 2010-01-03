####################################################################
# The Falcon Programming language
#
# CMake configuration file for module oracle
####################################################################

FALCON_DEFINE_MODULE( MODULE oracle )

# Find Oracle
SET(ORACLE_INCLUDE_SEARCH_PATH
  ${ORACLE_INCLUDE_DIR}
  /usr/lib/oracle
)
FIND_PATH(ORACLE_REAL_INCLUDE_DIR occi.h ${ORACLE_INCLUDE_SEARCH_PATH})

IF (MYSQL_REAL_INCLUDE_DIR)
  INCLUDE_DIRECTORIES(BEFORE ${ORACLE_REAL_INCLUDE_DIR})
  IF(NOT ORACLE_REAL_INCLUDE_DIR)
    MESSAGE(SEND_ERROR "Oracle include path was not found")
    MESSAGE(SEND_ERROR "Looked in: ${ORACLE_INCLUDE_SEARCH_PATH}")
  ENDIF (NOT ORACLE_REAL_INCLUDE_DIR)
ENDIF (ORACLE_REAL_INCLUDE_DIR)

MESSAGE(STATUS "Found occi.h in ${ORACLE_REAL_INCLUDE_DIR}")

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)

INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES("${FALCON_LIB_PATH}")

# Target
ADD_LIBRARY( ${MODULE} MODULE
   oracle.cpp
   oracle_ext.cpp
   oracle_srv.cpp
)

# Link
IF(MSVC)
  SET(ORACLE_OPT_LIBS wsock32)
  SET_TARGET_PROPERTIES(${MODULE}  PROPERTIES LINK_FLAGS "/NODEFAULTLIB:library")
ENDIF(MSVC)

TARGET_LINK_LIBRARIES( ${MODULE}  falcon_engine ${ORACLE_OPT_LIBS} ${ORACLE_LIBRARY})
FALCON_INSTALL_MODULE( ${MODULE}  )
