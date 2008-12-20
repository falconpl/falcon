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

INCLUDE_DIRECTORIES("${FALCON_INC_PATH}")
#INCLUDE_DIRECTORIES("${DBUS_INCLUDE_DIRS}")
ADD_DEFINITIONS(${DBUS_CFLAGS})

LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

#sources
FILE( GLOB SRC_FILES "*.cpp" )

# Target
ADD_LIBRARY( dbus MODULE
   ${SRC_FILES}
)

TARGET_LINK_LIBRARIES( dbus falcon_engine ${DBUS_LIBRARIES} )
FALCON_INSTALL_MODULE( dbus )
