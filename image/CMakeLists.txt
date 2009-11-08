####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_image
####################################################################

FALCON_DEFINE_MODULE( SDLIMAGE sdlimage )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)

Find_Package ( SDL_image REQUIRED )
INCLUDE_DIRECTORIES( ${SDLIMAGE_INCLUDE_DIR} )

# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${FALCON_INC_PATH})
LINK_DIRECTORIES(BEFORE ${FALCON_LIB_PATH})

# Target
ADD_LIBRARY( ${SDLIMAGE} MODULE
   sdlimage.cpp
   sdlimage_ext.cpp
   sdlimage_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(${SDLIMAGE} falcon_engine ${SDLIMAGE_LIBRARY} ${SDL_LIBRARY} )

SET_TARGET_PROPERTIES(${SDLIMAGE}
   PROPERTIES PREFIX "")

#Install
FALCON_LINK_MODULE( ${SDLIMAGE} )
