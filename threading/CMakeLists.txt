####################################################################
# The Falcon Programming language
#
# CMake configuration file for module threading
####################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${falcon_INCLUDE_DIRS}
)

#Specific system files
if(UNIX OR MAC)
   SET( SYS_SPECIFIC
      systhread_posix.cpp
    )
elseif(WIN32)
   set( SYS_SPECIFIC
      systhread_win.cpp
    )
endif()

add_library(threading_fm MODULE
   waitable.cpp
   threading.cpp
   threading_ext.cpp
   threading_mod.cpp
   threading_st.cpp
   ${SYS_SPECIFIC}
)

falcon_finalize_module( threading_fm )
 
