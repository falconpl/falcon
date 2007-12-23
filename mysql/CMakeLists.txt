####################################################################
# The Falcon Programming language
#
# CMake configuration file for module mysql
####################################################################
PROJECT(mysql)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

# Target
ADD_LIBRARY( mysql MODULE
   mysql.cpp
   mysql_ext.cpp
   mysql_srv.cpp
)

#Link
TARGET_LINK_LIBRARIES(mysql falcon_engine)
SET_TARGET_PROPERTIES(mysql
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( mysql )
