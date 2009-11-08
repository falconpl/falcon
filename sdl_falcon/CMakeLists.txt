##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################

# Inclusion settings
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${FALCON_INC_PATH}
)
LINK_DIRECTORIES(${FALCON_LIB_PATH})

ADD_EXECUTABLE( sdl_falcon
   falcon.cpp
   options.cpp
   int_mode.cpp
   SDLMain.m
)

#Link
TARGET_LINK_LIBRARIES(sdl_falcon falcon_engine ${SDL_LIBRARY})

   IF ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")
      SET( FALCON_BIN_PATH "$ENV{FALCON_BIN_PATH}" )
   ELSE ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")
      SET( FALCON_BIN_PATH "$ENV{FALCON_ACTIVE_TREE}/bin" )
   ENDIF ("$ENV{FALCON_ACTIVE_TREE}" STREQUAL "")

set(BINDEST ${INSTDIR})

if("${BINDEST}" STREQUAL "" )
   set(BINDEST ${FALCON_BIN_PATH})
endif("${BINDEST}" STREQUAL ""  )

#Install
MESSAGE( "Intalling sdl_falcon on ${BINDEST}" )
INSTALL( TARGETS sdl_falcon
      DESTINATION ${BINDEST} )
