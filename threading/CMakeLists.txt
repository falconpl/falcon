####################################################################
# The Falcon Programming language
#
# CMake configuration file for module threading
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

#Specific system files
IF(UNIX OR MAC)
   SET( SYS_SPECIFIC
      systhread_posix.cpp
    )

ELSEIF(WIN32)

   SET( SYS_SPECIFIC
      systhread_win.cpp
    )

ENDIF(UNIX OR MAC)

# Target
ADD_LIBRARY( threading_fm MODULE
   mt.cpp
   threading.cpp
   threading_ext.cpp
   threading_mod.cpp
   threading_st.cpp
   smembuf.cpp
   ${SYS_SPECIFIC}
)

#Link
IF(UNIX OR MAC)
   TARGET_LINK_LIBRARIES(threading_fm falcon_engine pthread )
ELSEIF(WIN32)
   TARGET_LINK_LIBRARIES(threading_fm falcon_engine )
ENDIF(UNIX OR MAC)

FALCON_INSTALL_MODULE( threading_fm )
