####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Mxml
####################################################################
PROJECT(sdl)

# Inclusion settings
add_definitions(`sdl-config --cflags`)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )

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
link_libraries(`sdl-config --libs`)
TARGET_LINK_LIBRARIES(sdl falcon_engine SDL)

SET_TARGET_PROPERTIES(sdl
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( sdl )
