####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_ttf
####################################################################
PROJECT(sdlmixer)

Find_Package ( SDL_mixer REQUIRED ) # if using SDL_mixer

# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")

# Target
ADD_LIBRARY( sdlmixer MODULE
   sdlmixer.cpp
   sdlmixer_ext.cpp
   sdlmixer_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(sdlmixer falcon_engine ${SDLMIXER_LIBRARY})

SET_TARGET_PROPERTIES(sdlmixer
   PROPERTIES PREFIX "")
#Install
FALCON_LINK_MODULE( sdlmixer )
