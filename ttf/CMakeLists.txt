####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_ttf
####################################################################
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
ADD_LIBRARY( sdlttf MODULE
   sdlttf.cpp
   sdlttf_ext.cpp
   sdlttf_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(sdlttf falcon_engine ${SDLTTF_LIBRARY} ${SDL_LIBRARY})

SET_TARGET_PROPERTIES(sdlttf
   PROPERTIES PREFIX "")
#Install
FALCON_LINK_MODULE( sdlttf )
