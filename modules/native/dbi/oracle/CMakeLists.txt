####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Oracle
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

falcon_define_module( FALCON_MODULE oracle )

SET(ORACLE_HOME $ENV{ORACLE_HOME})

SET(ORACLE_INCLUDE_SEARCH_PATH
    ${ORACLE_INCLUDE_DIR}
    ${ORACLE_HOME}/sdk/include
    ${ORACLE_HOME}/xdk/include
    ${ORACLE_HOME}/OCI/include
    ${ORACLE_HOME}/sdk/lib/msvc/
    ${ORACLE_HOME}/include
    /usr/local/include/oracle
)
FIND_PATH( ORACLE_INCLUDE_DIR occi.h ${ORACLE_INCLUDE_SEARCH_PATH} )

SET(ORACLE_LIB_SEARCH_PATH 
    ${ORACLE_LIB_DIR}
    ${ORACLE_HOME}/lib
    ${ORACLE_HOME}/OCI/lib/MSVC
    ${ORACLE_HOME}
    /usr/local/lib/oracle
)
FIND_LIBRARY( ORACLE_LIBRARY_OCCI NAMES libocci occi ${ORACLE_LIB_SEARCH_PATH} )
FIND_LIBRARY( ORACLE_LIBRARY_CLNTSH NAMES libclntsh clntsh ${ORACLE_LIB_SEARCH_PATH} )          

IF(APPLE)
    FIND_LIBRARY( ORACLE_LIBRARY_OCIEI NAMES libociei ociei ${ORACLE_LIB_SEARCH_PATH} )
ENDIF(APPLE)

IF(ORACLE_INCLUDE_DIR AND ORACLE_LIBRARY_OCCI AND ORACLE_LIBRARY_CLNTSH)
    INCLUDE_DIRECTORIES(BEFORE ${ORACLE_INCLUDE_DIR})
ELSE(ORACLE_INCLUDE_DIR AND ORACLE_LIBRARY)
    IF(NOT ORACLE_INCLUDE_DIR)
        MESSAGE(SEND_ERROR "Oracle include path was not found")
        MESSAGE(SEND_ERROR "Looked in: ${ORACLE_INCLUDE_SEARCH_PATH}")
    ENDIF (NOT ORACLE_INCLUDE_DIR)
    IF (NOT ORACLE_LIBRARY_OCCI )
        MESSAGE(SEND_ERROR "Oracle client library libocci was not found")
        MESSAGE(SEND_ERROR "Looked in: ${ORACLE_LIB_SEARCH_PATH}")
    ENDIF (NOT ORACLE_LIBRARY_OCCI)
    IF (NOT ORACLE_LIBRARY_CLNTSH )
        MESSAGE(SEND_ERROR "Oracle client library libclntsh was not found")
        MESSAGE(SEND_ERROR "Looked in: ${ORACLE_LIB_SEARCH_PATH}")
    ENDIF (NOT ORACLE_LIBRARY_CLNTSH)
    IF (NOT ORACLE_LIBRARY_OCIEI AND APPLE)
        MESSAGE(SEND_ERROR "Oracle client library libociei was not found")
        MESSAGE(SEND_ERROR "Looked in: ${ORACLE_LIB_SEARCH_PATH}")
    ENDIF (NOT ORACLE_LIBRARY_OCIEI AND APPLE)
    MESSAGE(FATAL_ERROR "Cannot build Oracle DBI module")
ENDIF(ORACLE_INCLUDE_DIR AND ORACLE_LIBRARY_OCCI AND ORACLE_LIBRARY_CLNTSH)

MESSAGE(STATUS "Found occi.h in ${ORACLE_INCLUDE_DIR}")
MESSAGE(STATUS "Found library clntsh ${ORACLE_LIBRARY_CLNTSH}")
MESSAGE(STATUS "Found library occi ${ORACLE_LIBRARY_OCCI}")

IF(APPLE)
    MESSAGE(STATUS "Found library ociei ${ORACLE_LIBRARY_OCIEI}")
ENDIF(APPLE)

# Inclusion settings
INCLUDE_DIRECTORIES(${ORACLE_INCLUDE_DIR})
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(../dbi)
INCLUDE_DIRECTORIES(.)

# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE
    
    ${dbi_common_files}

    oracle.cpp
    oracle_ext.cpp
    oracle_mod.cpp
)

# Link
TARGET_LINK_LIBRARIES( ${FALCON_MODULE} 
    falcon_engine 
    ${ORACLE_LIBRARY} 
)

falcon_install_module2( ${FALCON_MODULE} dbi )

