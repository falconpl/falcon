####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_ttf
####################################################################
PROJECT(sdlttf)

Find_Package ( SDL_ttf REQUIRED ) # if using SDL_image

# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")

# Target
ADD_LIBRARY( sdlttf MODULE
   sdlttf.cpp
   sdlttf_ext.cpp
   sdlttf_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(sdlttf falcon_engine ${SDLTTF_LIBRARY})

SET_TARGET_PROPERTIES(sdlttf
   PROPERTIES PREFIX "")
#Install
FALCON_LINK_MODULE( sdlttf )
