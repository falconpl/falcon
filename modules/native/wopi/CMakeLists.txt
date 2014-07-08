######################################################################
# CMake file for WOPI system
# Falcon Web Oriented Programming Interface
#

cmake_minimum_required(VERSION 2.6.2)

if(NOT FALCON_BUILD_WOPI )
	message( STATUS "NOT building WOPI subsystem")
	return()
endif()
message( STATUS "Building module WOPI subsystem" )

project( falcon_wopi ) 

######################################################################
# Build configuration
#

option( FALCON_WOPI_BUILD_FALCGI "Build Falcon CGI support" OFF )
option( FALCON_WOPI_BUILD_CGI_FM "Build Falcon module for stand-alone CGI scripts" OFF )
#option( FALCON_WOPI_BUILD_FFALCGI "Build Falcon FastCGI support" OFF )
option( FALCON_WOPI_BUILD_FALHTTPD "Build Falcon simple HTTP server" OFF )
option( FALCON_WOPI_BUILD_MOD_FALCON "Build Falcon module for Apache2" OFF )

######################################################################
# Required elements
#

find_package(Falcon
   REQUIRED)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${Falcon_INCLUDE_DIRS}
)

######################################################################
# Modules
#

add_subdirectory( include )
add_subdirectory( src )

# Mod falcon for apache 2
if( FALCON_WOPI_BUILD_MOD_FALCON )
   set(FALCON_WOPI_APACHE2_LIBEXECDIR "lib/apache2/modules" CACHE PATH "The location to install apache2 modules")
   add_subdirectory( apache2 )
   set(FALCON_WOPI_ONE_BUILD TRUE)
endif()

# CGI driver for falcon.
if( FALCON_WOPI_BUILD_FALCGI OR FALCON_WOPI_BUILD_FFALCGI )  
   set(FALCON_WOPI_CGI_INSTALL_DIR "lib/falcon/wopi/cgi-bin" CACHE PATH "The install location for CGI drivers")
endif()

#if( FALCON_WOPI_BUILD_FALCGI )
#   add_subdirectory( falcgi )
#   set(FALCON_WOPI_ONE_BUILD TRUE)
#endif()

if( FALCON_WOPI_BUILD_CGI_FM )
   add_subdirectory( cgi_fm )
   set(FALCON_WOPI_ONE_BUILD TRUE)
endif()

# FastCGI driver for falcon
if( FALCON_WOPI_BUILD_FFALCGI )
   add_subdirectory( ffalcgi )
   set(FALCON_WOPI_ONE_BUILD TRUE)
endif()

# Simple web server for Falcon.
if( FALCON_WOPI_BUILD_FALHTTPD )
   add_subdirectory( falhttpd )
   set(FALCON_WOPI_ONE_BUILD TRUE)
endif()

if (NOT FALCON_WOPI_ONE_BUILD)
   Message( FATAL_ERROR "Please, specify one or more of the following:
      -DFALCON_WOPI_BUILD_MOD_FALCON=ON
      -DFALCON_WOPI_BUILD_FALCGI=ON
      -DFALCON_WOPI_BUILD_FFALCGI=ON
      -DFALCON_WOPI_BUILD_CGI_FM=ON
      -DFALCON_WOPI_BUILD_FALHTTPD=ON
      ")
endif()
      
