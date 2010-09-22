####################################################################
# @MAIN_PRJ@
#
# CMake configuration file for @PROJECT_NAME@
####################################################################

# Creates the proper module file name from the project name.
falcon_define_module( CURRENT_MODULE conbox )

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Falcon_INCLUDE_DIRS}
)
# Enable this include if the project has a private include
include_directories( ${PROJECT_SOURCE_DIR}/include )


# Sources files the module is built from.
set(SRC_FILES 
  conbox.cpp
  conbox_ext.cpp
  conbox_st.cpp
  # add more source files starting here:
)

# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  conbox_ext.h
  conbox_st.h
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
