####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_image
####################################################################

FALCON_DEFINE_MODULE( SDLIMAGE sdlimage )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)

Find_Package ( SDL_image REQUIRED )

# Inclusion settings
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SDLIMAGE_INCLUDE_DIR} 
)

# Target
ADD_LIBRARY( ${SDLIMAGE} MODULE
   sdlimage.cpp
   sdlimage_ext.cpp
   sdlimage_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(${SDLIMAGE} falcon_engine ${SDLIMAGE_LIBRARY} ${SDL_LIBRARY} )

SET_TARGET_PROPERTIES(${SDLIMAGE}
   PROPERTIES PREFIX "")

#Install
FALCON_INSTALL_MODULE( ${SDLIMAGE} )
