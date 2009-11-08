####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL
####################################################################

FALCON_DEFINE_MODULE( MODSDL sdl )

 if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
 endif(COMMAND cmake_policy)

# Inclusion settings
INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${FALCON_INC_PATH}
  ${FALCON_LIB_PATH}
)

# Target
ADD_LIBRARY( ${MODSDL} MODULE
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
TARGET_LINK_LIBRARIES(${MODSDL} falcon_engine ${SDL_LIBRARY})

SET_TARGET_PROPERTIES(${MODSDL}
   PROPERTIES PREFIX "")

#Install
FALCON_LINK_MODULE( ${MODSDL} )
