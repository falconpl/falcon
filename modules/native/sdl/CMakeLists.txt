####################################################################
# The Falcon Programming language
#
# CMake configuration file for SDL
####################################################################
if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)
cmake_minimum_required(VERSION 3.16.3)
PROJECT(Falcon_SDL)


# REQUIRED does not work in CMake <=2.4.6 for SDL
Find_Package ( SDL REQUIRED )

# Workaround for the non-working REQUIRED flag
if ( NOT SDL_FOUND )
   message ( FATAL_ERROR "SDL not found!" )
endif ( NOT SDL_FOUND )

find_package(Falcon REQUIRED)

INCLUDE_DIRECTORIES( 
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${Falcon_INCLUDE_DIRS}
   ${SDL_INCLUDE_DIR} 
)

option(FALCON_SDL_OPENGL "Build also GL, GLX and GLU modules" OFF)

###############################àà
# Echo config
Message( "OpenGL support: ${FALCON_SDLOPENGL}" )

# Project directories
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(ttf)
ADD_SUBDIRECTORY(image)
ADD_SUBDIRECTORY(mixer)

IF( FALCON_SDL_OPENGL )
ADD_SUBDIRECTORY(gl)
ADD_SUBDIRECTORY(glu)
ADD_SUBDIRECTORY(glext)
ENDIF()

#Add the sdl_falcon framework executable on macosx
if(APPLE)
   ADD_SUBDIRECTORY(sdl_falcon)
endif(APPLE)
