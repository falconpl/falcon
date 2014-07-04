###############################################################################
# Falcon Programming language
# Build and installation suite
#
# VFS modules
#

if(NOT FALCON_BUILD_VFS )
	message( STATUS "NOT building VFS subsystem")
	return()
endif()

message( STATUS "Building VFS subsystem")



add_subdirectory( zip )
