###############################################################################
# Falcon Programming language
# Build and installation suite
#
# VFS modules
#

OPTION( FALCON_BUILD_ZIP "Install Zip module" ON )

if ( FALCON_BUILD_ZIP )
    add_subdirectory( zip )
endif()