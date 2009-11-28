####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

FALCON_DEFINE_MODULE( CURRENT_MODULE curl )

# Enable this include if the project has a private include
#INCLUDE_DIRECTORIES( "${PROJECT_SOURCE_DIR}/include" )

#
# Falcon generic directories
#

INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
INCLUDE_DIRECTORIES(BEFORE "${CURL_INCLUDE_DIR}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}" )

#sources
FILE( GLOB SRC_FILES "*.cpp" "*.h" )

# Target
ADD_LIBRARY( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
)

SET_TARGET_PROPERTIES(${CURRENT_MODULE} PROPERTIES LINKER_LANGUAGE CXX)

TARGET_LINK_LIBRARIES( ${CURRENT_MODULE} falcon_engine)
TARGET_LINK_LIBRARIES( ${CURRENT_MODULE} "${CURL_LIBRARY}")
FALCON_INSTALL_MODULE( ${CURRENT_MODULE} )
#
