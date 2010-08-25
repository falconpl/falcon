####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for MP
####################################################################



# Inclusion settings
INCLUDE_DIRECTORIES(.)
# Enable this include if the project has a private include
#INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )

#
# Falcon generic directories
#

INCLUDE_DIRECTORIES( ${falcon_INCLUDE_DIRS})
LINK_DIRECTORIES("${FALCON_LIB_PATH}")

#sources
FILE( GLOB SRC_FILES "*.cpp" )

# Target
ADD_LIBRARY( MP_fm MODULE
   ${SRC_FILES}
)

#Link
TARGET_LINK_LIBRARIES( MP_fm ${GMP_LIBRARIES})
falcon_finalize_module( MP_fm )

#If your module needs some lib more, use the followings:
#
# TARGET_LINK_LIBRARIES( ${CURRENT_MODULE} falcon_engine)
# FALCON_INSTALL_MODULE( ${CURRENT_MODULE} )
#

