####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_image
####################################################################
PROJECT(sdlimage)

Find_Package ( SDL_image REQUIRED ) # if using SDL_image

# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")

# Target
ADD_LIBRARY( sdlimage MODULE
   sdlimage.cpp
   sdlimage_ext.cpp
   sdlimage_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(sdlimage falcon_engine ${SDLIMAGE_LIBRARY})

SET_TARGET_PROPERTIES(sdlimage
   PROPERTIES PREFIX "")

#Install
FALCON_LINK_MODULE( sdlimage )
