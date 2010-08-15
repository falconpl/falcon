####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dbus
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)
# Enable this include if the project has a private include
#INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )

#
# Falcon generic directories
#

find_package( Falcon REQUIRED )
INCLUDE_DIRECTORIES( ${Falcon_INCLUDE_DIRS} )
#INCLUDE_DIRECTORIES("${DBUS_INCLUDE_DIRS}")
ADD_DEFINITIONS(${DBUS_CFLAGS})

LINK_DIRECTORIES( ${FALCON_LIB_PATH} )

#sources
FILE( GLOB SRC_FILES "*.cpp" )

# Target
ADD_LIBRARY( dbus_fm MODULE
   ${SRC_FILES}
)

TARGET_LINK_LIBRARIES( dbus_fm falcon_engine ${DBUS_LIBRARIES} )
FALCON_INSTALL_MODULE( dbus_fm )
