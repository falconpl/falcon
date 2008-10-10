##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

ADD_EXECUTABLE( sdl_falcon
   falcon.cpp
   SDLMain.m
)

#Link
TARGET_LINK_LIBRARIES(sdl_falcon falcon_engine ${SDL_LIBRARY})

   IF ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")
      SET( FALCON_BIN_PATH "$ENV{FALCON_BIN_PATH}" )
   ELSE ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")
      SET( FALCON_BIN_PATH "$ENV{FALCON_ACTIVE_TREE}/bin" )
   ENDIF ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")

#Install
INSTALL( TARGETS sdl_falcon
      DESTINATION ${FALCON_BIN_PATH} )
