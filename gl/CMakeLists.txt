####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_ttf
####################################################################

FALCON_DEFINE_MODULE( SDLGL sdlgl )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)

Find_Package ( SDL REQUIRED )
Find_package ( OpenGL REQUIRED )

# Inclusion settings
INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${FALCON_INC_PATH}
  ${FALCON_LIB_PATH}
  ${SDL_INCLUDE_DIR}
)

# Target
ADD_LIBRARY( ${SDLGL} MODULE
   gl.cpp
   gl_ext.cpp
   gl_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES( ${SDLGL} falcon_engine ${SDL_LIBRARY} ${OPENGL_LIBRARY})

SET_TARGET_PROPERTIES( ${SDLGL}
   PROPERTIES PREFIX "")
#Install
FALCON_LINK_MODULE( ${SDLGL} )
