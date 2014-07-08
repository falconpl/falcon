# FalconFindDataMatrix.cmake
#
# Set vars _dmtx to ON or OFF,
# DMTX_INCLUDE_PATH and DMTX_LIBRARY.

if ( WIN32 )
    find_path( DMTX_INCLUDE_PATH dmtx.h
        HINTS
            c:/dmtx
            c:/libdmtx-0.7.2
            c:/data/dmtx
            c:/data/libdmtx-0.7.2
        )
    find_library( DMTX_LIBRARY libdmtx
        HINTS
            c:/dmtx
            c:/libdmtx-0.7.2
            c:/data/dmtx
            c:/data/libdmtx-0.7.2
        )
else()
    find_path( DMTX_INCLUDE_PATH dmtx.h
        HINTS /usr/include /usr/local/include )
    find_library( DMTX_LIBRARY dmtx
        HINTS /usr/lib /usr/local/lib )
endif()

mark_as_advanced( DMTX_INCLUDE_PATH )
mark_as_advanced( DMTX_LIBRARY )

if ( ( DMTX_INCLUDE_PATH STREQUAL DMTX_INCLUDE_PATH-NOTFOUND )
       OR ( DMTX_LIBRARY STREQUAL DMTX_LIBRARY-NOTFOUND ) )
    set( _dmtx OFF )
else()
    set( _dmtx ON )
endif()
