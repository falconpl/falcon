###############################################################################
# Falcon Programming language
# Build and installation suite
#
# Native modules
#

#########################################################
# Utilities used in this section
#
macro( FalconMakeBinding module lib desc )
   option( FALCON_BUILD_${module} ${desc} OFF )

   if( FALCON_NATMODS_AUTO )
      find_package(${lib})
      if ( ${lib}_FOUND )
         set( FALCON_BUILD_${module} ON )
      endif()
   endif()
endmacro()

#########################################################


OPTION( FALCON_NATMODS_AUTO "Select automatically all the modules that have support" ON )

# Native modules, usually compiled
OPTION( FALCON_BUILD_DBI "Install DBI module" ON )
OPTION( FALCON_BUILD_DYNLIB "Install DYNLIB module" ON )

# Bindings -- native modules requiring external libraries.
FalconMakeBinding( CURL CURL "Install CURL binding" )
FalconMakeBinding( GTK GTK2 "Install GTK binding" )
FalconMakeBinding( SDL SDL "Install SDL  binding" )

# Bindings -- Requiring more sophisticated checks
option( FALCON_BUILD_DBUS "Install DBUS binding" OFF )
option( FALCON_BUILD_GD2 "Install GD library binding" OFF )
option( FALCON_BUILD_PDF "Install libharu (pdf)  binding" OFF )

if( FALCON_NATMODS_AUTO )

   #### Check DBUS
   pkg_check_modules( DBUS dbus-1 )
   if (DBUS_INCLUDE_DIRS)
      set( FALCON_BUILD_DBUS ON )
   endif()

   #### Check LibHARU
   find_path(LibHaru_INCLUE_DIR hpdf.h)
   if (LibHaru_INCLUE_DIR)
      set( FALCON_BUILD_PDF ON )
   endif()

   #### Check GD2
   find_path(GD_INCLUDE_DIR gd.h)
   if (GD_INCLUDE_DIR)
      set( FALCON_BUILD_GD2 ON )
   endif()

endif()

#########################################################
# Inclusion statements
#

if(FALCON_BUILD_CURL)
   add_subdirectory( curl )
endif()

if ( FALCON_BUILD_DBI )
    add_subdirectory( dbi )
endif()

if ( FALCON_BUILD_DBUS )
    add_subdirectory( dbus )
endif()

if ( FALCON_BUILD_DYNLIB )
    add_subdirectory( dynlib )
endif()

if ( FALCON_BUILD_GD2 )
    add_subdirectory( gd2 )
endif()

if ( FALCON_BUILD_GTK )
    add_subdirectory( gtk )
endif()

if ( FALCON_BUILD_PDF )
    add_subdirectory( pdf )
endif()

if ( FALCON_BUILD_SDL )
    add_subdirectory( sdl )
endif()



