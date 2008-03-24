####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Mxml
####################################################################
PROJECT(sdlttf)

# Inclusion settings
add_definitions(`sdl-config --cflags`)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )

# Target
ADD_LIBRARY( sdlttf MODULE
   sdlttf.cpp
   sdlttf_ext.cpp
   sdlttf_mod.cpp
)

#Link
link_libraries(`sdl-config --libs`)
TARGET_LINK_LIBRARIES(sdlttf falcon_engine SDL_ttf)

SET_TARGET_PROPERTIES(sdlttf
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( sdlttf )
