####################################################################
# The Falcon Programming language
#
# CMake configuration file for module sqlite3
####################################################################
PROJECT(sqlite3)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )

# Target
ADD_LIBRARY( sqlite3 MODULE
   sqlite3.cpp
   sqlite3_ext.cpp
   sqlite3_srv.cpp
   sqlite.c
)

#Link
TARGET_LINK_LIBRARIES(sqlite3 falcon_engine pthread)
SET_TARGET_PROPERTIES(sqlite3 PROPERTIES PREFIX "") #Install
FALCON_MODULE_INSTALL( sqlite3 )
