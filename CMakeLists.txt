####################################################################
# The Falcon Programming language
#
# CMake configuration file for SDL
####################################################################
cmake_minimum_required(VERSION 2.4)
PROJECT(Falcon_SDL)

# Check for environment
INCLUDE( "falmod.cmake" )

#Find SDL if SDL_INCLUDE_PATH has not specified.
IF(NOT SDL_INCLUDE_PATH)
   IF(NOT SDL_INCLUDE_SEARCH_PATH)
      SET(SDL_INCLUDE_SEARCH_PATH /usr/include /usr/local/include )
   ENDIF(NOT SDL_INCLUDE_SEARCH_PATH)

   FIND_PATH(SDL_INCLUDE_PATH sdl.h ${SDL_INCLUDE_SEARCH_PATH})
ENDIF(NOT SDL_INCLUDE_PATH) 


#Find SDL if SDL_LIB_PATH has not specified.
IF(NOT SDL_LIB_PATH)
   IF(NOT SDL_LIB_SEARCH_PATH)
      SET(SDL_LIB_SEARCH_PATH /usr/lib64 /usr/local/lib64 /usr/lib /usr/local/lib)
   ENDIF(NOT SDL_LIB_SEARCH_PATH)

   IF(WIN32)
      FIND_PATH(SDL_LIB_PATH sdl.lib ${SDL_INCLUDE_SEARCH_PATH})
   ELSE(WIN32)
      FIND_PATH(SDL_LIB_PATH libsdl.a ${SDL_INCLUDE_SEARCH_PATH})
      IF( NOT SDL_LIB_PATH )
         FIND_PATH(SDL_LIB_PATH libsdl.sdl ${SDL_INCLUDE_SEARCH_PATH})
      ENDIF( NOT SDL_LIB_PATH )
   ENDIF(WIN32)
ENDIF(NOT SDL_LIB_PATH) 



IF(SDL_INCLUDE_PATH AND SDL_LIB_PATH)
   
   # Setting project wide includes and libs
   INCLUDE_DIRECTORIES(${SDL_INCLUDE_PATH})
   INCLUDE_DIRECTORIES(${SDL_REAL_INCLUDE_PATH})

   # Project directories
   ADD_SUBDIRECTORY(src)
   ADD_SUBDIRECTORY(ttf)
   ADD_SUBDIRECTORY(image)

ELSE(SDL_INCLUDE_PATH AND SDL_LIB_PATH)
   IF(NOT SDL_INCLUDE_PATH)
      MESSAGE(SEND_ERROR "SDL include path was not found")
      MESSAGE(SEND_ERROR "Looked in: ${SDL_INCLUDE_SEARCH_PATH}")
   ENDIF(NOT SDL_INCLUDE_PATH)
   
   IF(NOT SDL_LIB_PATH)
      MESSAGE(SEND_ERROR "SDL library was not found")
      MESSAGE(SEND_ERROR "Looked in: ${SDL_LIB_SEARCH_PATH}")
   ENDIF(NOT SDL_LIB_PATH)
   
   MESSAGE(SEND_ERROR "Could not find SDL, you can specify:")
   MESSAGE(SEND_ERROR "-DSDL_INCLUE_PATH=PATH_TO_SDL_INCLUDES")
   MESSAGE(FATAL_ERROR "-DSDL_LIB_PATH=PATH_TO_SDL_LIBRARY")
ENDIF(SDL_INCLUDE_PATH AND SDL_LIB_PATH)
