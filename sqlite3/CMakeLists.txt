####################################################################
# The Falcon Programming language
#
# CMake configuration file for module sqlite3
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FALCON_DEFINE_MODULE( FALCON_MODULE sqlite3 )

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)

# Target
ADD_LIBRARY( ${FALCON_MODULE} MODULE
   sqlite3.cpp
   sqlite3_ext.cpp
   sqlite3_srv.cpp
   sqlite.c
)

#Link

IF(WIN32)
   TARGET_LINK_LIBRARIES(${FALCON_MODULE} falcon_engine)
ELSE(WIN32)
   #Non-win32 builds require pthreads
   TARGET_LINK_LIBRARIES(${FALCON_MODULE} falcon_engine pthread)
ENDIF(WIN32)

falcon_install_module( ${FALCON_MODULE} )
