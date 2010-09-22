####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for conio
####################################################################


# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE conio )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)
# Enable this include if the project has a private include
include_directories( ${PROJECT_SOURCE_DIR}/include )

# Sources files the module is built from.
set(SRC_FILES 
  conio.cpp
  conio_ext.cpp
  conio_mod.cpp
  conio_st.cpp
  # add more source files starting here:
)

#system specific sources
if(WIN32)
  list(APPEND SRC_FILES sys/conio_srv_win.cpp)
endif()

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  conio_ext.h
  conio_mod.h
  conio_st.h
  version.h
  # add more header files starting here:
)


# Builds and links from the source files
add_library( ${CURRENT_MODULE} MODULE
   ${SRC_FILES}
   ${HDR_FILES} # optional, see comment above.
)

# If your module needs some lib more, use the followings:
target_link_libraries( ${CURRENT_MODULE} falcon_engine)

falcon_install_module( ${CURRENT_MODULE} )
