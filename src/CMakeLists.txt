####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Mxml
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")

# Target
ADD_LIBRARY( sdl MODULE
   sdl.cpp
   sdl_ext.cpp
   sdl_cursor_ext.cpp
   sdl_events_ext.cpp
   sdl_wm_ext.cpp
   sdl_surface_ext.cpp
   sdl_service.cpp
   sdl_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(sdl falcon_engine ${SDL_LIBRARY})

SET_TARGET_PROPERTIES(sdl
   PROPERTIES PREFIX "")

#Install
FALCON_LINK_MODULE( sdl )
