####################################################################
# FALCON - The Falcon Programming Language.
#
# Cgi module for FALCON scripts
####################################################################


MESSAGE( "Creating the Falcon CGI script module" )


#######################################################################
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
  
  cgi.cpp
  cgifm_ext.cpp
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  ../falcgi/cgi_make_streams.h
  ../falcgi/cgi_options.h
  ../falcgi/cgi_reply.h
  ../falcgi/cgi_request.h
  
  cgifm_ext.h
)

if(WIN32)
   set( RC_FILES cgi.rc )
endif()


##############################################
# Target

ADD_LIBRARY( cgi_fm MODULE
   ${SRC_FILES}
   ${WOPI_SOURCES}
   ${HDR_FILES} # optional but useful, see comment above.
   ${RC_FILES}
)
target_link_libraries(cgi_fm
   falcon_engine
)

# link to falcon_engine and install
falcon_install_module( cgi_fm )
