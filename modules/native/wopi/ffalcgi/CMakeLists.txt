######################################################################
# CMake file for fcgi
#

MESSAGE( "Creating the Falcon FastCGI driver" )

######################################################################
# Search FastCGI

SET( FASTCGI_INCLUDE_SEARCH_PATH
  ${FASTCGI_INCLUDE_DIR}
  /opt/include
  /usr/local/include
  /usr/include/
)

SET( FASTCGI_LIB_SEARCH_PATH
  ${FASTCGI_LIB_DIR}
  /opt/lib
  /usr/local/lib
  /usr/lib
  /opt/lib64
  /usr/local/lib64
  /usr/lib64
)

FIND_PATH(FASTCGI_REAL_INCLUDE_DIR fastcgi.h PATHS ${FASTCGI_INCLUDE_SEARCH_PATH} PATH_SUFFIXES fastcgi)
FIND_PATH(FASTCGI_REAL_LIB_DIR NAMES libfcgi.a libfcgi.so PATHS ${FASTCGI_LIB_SEARCH_PATH})

if(NOT FASTCGI_INCLUDE_SEARCH_PATH)
   Message(FATAL_ERROR "Can't find FastCGI include headers." )
endif()

if(NOT FASTCGI_REAL_LIB_DIR)
   Message(FATAL_ERROR "Can't find FastCGI libraries." )
endif()

INCLUDE_DIRECTORIES("${FASTCGI_REAL_INCLUDE_DIR}")
link_directories( "${FASTCGI_REAL_LIB_DIR}" )
Message( "Found FastCGI at ${FASTCGI_REAL_INCLUDE_DIR}" )

######################################################################
# Targets
#

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/falcgi
  ${Falcon_INCLUDE_DIRS}
)

set(SRC_FILES
  ../falcgi/cgi_options.cpp
  ../falcgi/cgi_reply.cpp
  ../falcgi/cgi_request.cpp
  ../falcgi/falcgi_perform.cpp
  
  ffalcgi.cpp
  ffalcgi_make_streams.cpp
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  ../falcgi/cgi_make_streams.h
  ../falcgi/cgi_options.h
  ../falcgi/cgi_reply.h
  ../falcgi/cgi_request.h
  ../falcgi/falcgi_perform.h
)

#INCLUDE_DIRECTORIES("../falcgi")

ADD_EXECUTABLE( ffalcgi
   ${SRC_FILES}
   ${WOPI_SOURCES}
   ${HDR_FILES} # optional but useful, see comment above.
)

if(WIN32)
	set( LIBFCGI libfcgi )
else()
	set( LIBFCGI fcgi )
endif()

TARGET_LINK_LIBRARIES( ffalcgi
   falcon_engine
   ${LIBFCGI}
   ) #APR library added in ld flags

INSTALL(TARGETS ffalcgi
   RUNTIME DESTINATION ${FALCON_WOPI_CGI_INSTALL_DIR}
)

