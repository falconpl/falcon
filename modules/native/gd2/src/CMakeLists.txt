####################################################################
# The Falcon Programming Language
#
# CMake configuration file for gd2
####################################################################

FALCON_DEFINE_MODULE( CURRENT_MODULE gd2 )

# Inclusion settings
INCLUDE_DIRECTORIES(.)

#
# Needed stuff
#

FIND_PACKAGE(GD REQUIRED)
MESSAGE( STATUS "Library GD found in: ${GD_LIBRARIES}" )


if(NOT FREETYPE_LIB)
FIND_LIBRARY( FREETYPE_LIB 
   NAMES libfreetype.so libfreetype.dylib freetype.lib 
   PATH /usr/lib /usr/local/lib ${FREETYPE_LIB_DIR}) 
endif()

if(NOT FONTCONFIG_LIB)
FIND_LIBRARY( FONTCONFIG_LIB 
   NAMES libfontconfig.so libfontconfig.dylib fontconfig.lib 
   PATH /usr/lib /usr/local/lib ${FONTCONFIG_LIB_DIR}) 
endif()

if(FREETYPE_LIB)
   MESSAGE( STATUS "Library freetype found in: ${FREETYPE_LIB}" )
else()
   MESSAGE( FATAL_ERROR "Can't find freetype library" ) 
endif()

if(FONTCONFIG_LIB)
   MESSAGE( STATUS "Library fontconfig found in: ${FONTCONFIG_LIB}" )
else()
   MESSAGE( FATAL_ERROR "Can't find fontconfig library" ) 
endif()

INCLUDE_DIRECTORIES( ${Falcon_INCLUDE_DIRS} ${GD_INCLUDE_DIRS})
LINK_DIRECTORIES( ${FALCON_LIB_PATH} )

#sources
FILE( GLOB SRC_FILES "*.cpp" )

# Target
ADD_LIBRARY( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
)

#Link
#FALCON_LINK_MODULE( ${CURRENT_MODULE} )

#If your module needs some lib more, use the followings:
#
TARGET_LINK_LIBRARIES( ${CURRENT_MODULE} falcon_engine ${GD_LIBRARIES} ${FONTCONFIG_LIB} ${FREETYPE_LIB} m)
FALCON_INSTALL_MODULE( ${CURRENT_MODULE} )
#
