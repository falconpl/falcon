# FalconFindGTK2.cmake
#
# Set var _gtk to ON or OFF.
# On Windows, set a special var GTK_BUNDLE_DIR.

if ( WIN32 )
    find_file( GTK_BUNDLE_DIR
        NAMES gtk gtk2 GTK GTK2
        HINTS c:/ c:/data ENV GTK_BUNDLE_DIR )
    if ( GTK_BUNDLE_DIR STREQUAL GTK_BUNDLE_DIR-NOTFOUND )
        set( _gtk OFF )
    else()
        set( _gtk ON )
    endif()
else()
    find_package( GTK2 )
    if ( GTK2_FOUND )
        set( _gtk ON )
    else()
        set( _gtk OFF )
    endif()
endif()
