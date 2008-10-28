####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)
# Enable this include if the project has a private include
#INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )

#
# Falcon generic directories
#

INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

#sources
FILE( GLOB SRC_FILES "*.cpp" )

# Target
ADD_LIBRARY( dynlib MODULE
   ${SRC_FILES}
)

#Link
FALCON_LINK_MODULE( dynlib )

#If your module needs some lib more, use the followings:
#
# TARGET_LINK_LIBRARIES( dynlib falcon_engine)
# FALCON_INSTALL_MODULE( dynlib )
#
