##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################

# Inclusion settings
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

ADD_EXECUTABLE( sdl_falcon
   falcon.cpp
   options.cpp
   int_mode.cpp
   SDLMain.m
)

#Link
TARGET_LINK_LIBRARIES(sdl_falcon falcon_engine ${SDL_LIBRARY})

#Install
MESSAGE( "Intalling sdl_falcon on ${BINDEST}" )
INSTALL( TARGETS sdl_falcon
      DESTINATION ${Falcon_BIN_DIR} )
