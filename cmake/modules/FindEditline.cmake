### FindEditLine.cmake module for Falcon

option( FALCON_WITH_GPL_READLINE "Insist on using GNU readline" OFF )

# unset these so that we can actually find GNU readline
unset( Editline_INCLUE_DIR CACHE )
unset( Editline_LIBRARY CACHE )

if( FALCON_WITH_GPL_READLINE )
  set( _header readline/readline.h )
  set( _libname readline )
else()
  set( _header editline/readline.h )
  set( _libname edit )
endif()

find_path( Editline_INCLUE_DIR ${_header} )
find_library( Editline_LIBRARY ${_libname} )

include( CheckSymbolExists )
set( CMAKE_REQUIRED_INCLUDES ${Editline_INCLUE_DIR} )
set( CMAKE_REQUIRED_LIBRARIES ${Editline_LIBRARY} )
check_symbol_exists( rl_getc_function "${_header}" HAVE_RL_GETC_FUNCTION )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Editline DEFAULT_MSG
  Editline_LIBRARY Editline_INCLUE_DIR HAVE_RL_GETC_FUNCTION )

set( Editline_INCLUDE_DIRS ${Editline_INCLUE_DIR} )
set( Editline_LIBRARIES ${Editline_LIBRARY} )

# vi: ts=2 sw=2 et
