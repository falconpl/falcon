####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_image
####################################################################
PROJECT(sdlimage)

# Inclusion settings
add_definitions(`sdl-config --cflags`)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_DIR}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_DIR}")

# Target
ADD_LIBRARY( sdlimage MODULE
   sdlimage.cpp
   sdlimage_ext.cpp
   sdlimage_mod.cpp
)

#Link
link_libraries(`sdl-config --libs`)
TARGET_LINK_LIBRARIES(sdlimage falcon_engine SDL_image)

SET_TARGET_PROPERTIES(sdlimage
   PROPERTIES PREFIX "")
#Install
FALCON_LINK_MODULE( sdlimage )
