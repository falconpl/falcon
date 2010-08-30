######################################################################
# CMake file for fcgi
#

MESSAGE( "Creating the Falcon CGI driver" )


#######################################################################
# Targets
#

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${Falcon_INCLUDE_DIRS}
)

# Sources files the library is built from.
set(SRC_FILES
  cgi_options.cpp
  cgi_reply.cpp
  cgi_request.cpp
  falcgi.cpp
  falcgi_make_streams.cpp
  falcgi_perform.cpp
)
# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  cgi_make_streams.h
  cgi_options.h
  cgi_reply.h
  cgi_request.h
  falcgi_perform.h
)

ADD_EXECUTABLE( falcgi
   ${SRC_FILES}
   ${WOPI_SOURCES}
   ${HDR_FILES} # optional but useful, see comment above.
)


TARGET_LINK_LIBRARIES( falcgi
   falcon_engine
   ${SYSLIBS}
   ) #APR library added in ld flags

INSTALL(TARGETS falcgi
  RUNTIME DESTINATION ${CGI_INSTALL_DIR}
)
