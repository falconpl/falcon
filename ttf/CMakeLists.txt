####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_ttf
####################################################################

FALCON_DEFINE_MODULE( SDLTTF sdlttf )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)

Find_Package ( SDL_ttf REQUIRED )
INCLUDE_DIRECTORIES( ${SDLTTF_INCLUDE_DIR} )

# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

# Target
ADD_LIBRARY( ${SDLTTF} MODULE
   sdlttf.cpp
   sdlttf_ext.cpp
   sdlttf_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES( ${SDLTTF} falcon_engine ${SDLTTF_LIBRARY} ${SDL_LIBRARY})

SET_TARGET_PROPERTIES( ${SDLTTF}
   PROPERTIES PREFIX "")
#Install
FALCON_LINK_MODULE( ${SDLTTF} )
