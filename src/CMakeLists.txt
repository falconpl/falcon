####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL
####################################################################

 if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
 endif(COMMAND cmake_policy)
 
# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

# Target
ADD_LIBRARY( fsdl MODULE
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
TARGET_LINK_LIBRARIES(fsdl falcon_engine ${SDL_LIBRARY}  )

if(APPLE)
   find_file( SDLMAIN_LIBRARY SDLMain.nib PATHS /Library/Frameworks/SDL.Framework/Resources )
   add_definitions( -l${SDLMAIN_LIBRARY} )
endif(APPLE)

SET_TARGET_PROPERTIES(fsdl
   PROPERTIES PREFIX "")

#Install
FALCON_LINK_MODULE( fsdl )
