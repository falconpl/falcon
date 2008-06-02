####################################################################
# The Falcon Programming language
#
# CMake configuration file for module sqlite3
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_DIR}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_DIR}")

# Target
ADD_LIBRARY( sqlite3 MODULE
   sqlite3.cpp
   sqlite3_ext.cpp
   sqlite3_srv.cpp
   sqlite.c
)

#Link

IF(WIN32)
   TARGET_LINK_LIBRARIES(sqlite3 falcon_engine)
ELSE(WIN32)
   #Non-win32 builds require pthreads
   TARGET_LINK_LIBRARIES(sqlite3 falcon_engine pthread)
ENDIF(WIN32)

FALCON_INSTALL_MODULE( sqlite3 )
