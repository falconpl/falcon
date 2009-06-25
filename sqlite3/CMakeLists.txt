####################################################################
# The Falcon Programming language
#
# CMake configuration file for module sqlite3
####################################################################

FALCON_DEFINE_MODULE( MODULE sqlite3 )

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES("${FALCON_INC_PATH}")
LINK_DIRECTORIES("${FALCON_LIB_PATH}")

# Target
ADD_LIBRARY( ${MODULE} MODULE
   sqlite3.cpp
   sqlite3_ext.cpp
   sqlite3_srv.cpp
   sqlite.c
)

#Link

IF(WIN32)
   TARGET_LINK_LIBRARIES(${MODULE} falcon_engine)
ELSE(WIN32)
   #Non-win32 builds require pthreads
   TARGET_LINK_LIBRARIES(${MODULE} falcon_engine pthread)
ENDIF(WIN32)

FALCON_INSTALL_MODULE( ${MODULE} )
