# Falcon Modules

if ( FALCON_WANT_CURL )
    add_subdirectory( curl )
endif()

if ( FALCON_WANT_DBI )
    add_subdirectory( dbi )
endif()

if ( FALCON_WANT_DBUS )
    add_subdirectory( dbus )
endif()

if ( FALCON_WANT_DYNLIB )
    add_subdirectory( dynlib )
endif()

if ( FALCON_WANT_GD2 )
    add_subdirectory( gd2 )
endif()

if ( FALCON_WANT_GTK )
    add_subdirectory( gtk )
endif()

if ( FALCON_WANT_PDF )
    add_subdirectory( pdf )
endif()

if ( FALCON_WANT_SDL )
    add_subdirectory( sdl )
endif()

