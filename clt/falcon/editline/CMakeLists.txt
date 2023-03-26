#/# Editline for Falcon

cmake_minimum_required( VERSION 3.16.3 )
project( Editline )

message( STATUS "Configuring Editline ..." )

find_package( Curses REQUIRED )
set( HAVE_CURSES_H ${CURSES_HAVE_CURSES_H} )
set( HAVE_NCURSES_H ${CURSES_HAVE_NCURSES_H} )

include( CheckIncludeFile )
include( CheckSymbolExists )
include( CheckTypeSize )

set( CMAKE_REQUIRED_DEFINITIONS
  -D_GNU_SOURCE # secure_getenv
)

# includes
check_include_file( stdint.h HAVE_STDINT_H )
check_include_file( sys/types.h HAVE_SYS_TYPES_H )
check_include_file( term.h HAVE_TERM_H )
check_include_file( termcap.h HAVE_TERMCAP_H )

# pwd.h
check_include_file( pwd.h HAVE_PWD_H )
if ( HAVE_PWD_H )
  # assuming posix.1, we're not checking for draft posix.1 version
  check_symbol_exists( getpwnam_r pwd.h HAVE_GETPWNAM_R )
  check_symbol_exists( getpwuid_r pwd.h HAVE_GETPWUID_R )
  if ( HAVE_GETPWNAM_R AND HAVE_GETPWUID_R )
    set( HAVE_GETPW_R_POSIX ON )
  endif()
endif()

# stdlib.h
check_include_file( stdlib.h HAVE_STDLIB_H )
if ( HAVE_STDLIB_H )
  check_symbol_exists( __secure_getenv stdlib.h HAVE___SECURE_GETENV )
  check_symbol_exists( reallocarr stdlib.h HAVE_REALLOCARR )
  check_symbol_exists( secure_getenv stdlib.h HAVE_SECURE_GETENV )
endif()

# string.h
check_include_file( string.h HAVE_STRING_H )
if ( HAVE_STRING_H )
  check_symbol_exists( strlcat string.h HAVE_STRLCAT )
  check_symbol_exists( strlcpy string.h HAVE_STRLCPY )
endif()

# unistd.h
check_include_file( unistd.h HAVE_UNISTD_H )
if ( HAVE_UNISTD_H )
  check_symbol_exists( issetugid unistd.h HAVE_ISSETUGID )
endif()

# vis.h
check_include_file( vis.h HAVE_VIS_H )
if ( HAVE_VIS_H )
  check_symbol_exists( vis vis.h HAVE_VIS )
endif()

# wchar.h
check_include_file( wchar.h HAVE_WCHAR_H )
if ( HAVE_WCHAR_H )
  check_symbol_exists( wcsdup wchar.h HAVE_WCSDUP )
endif()

# type checks
check_type_size( u_int32_t U_INT32_T )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

add_subdirectory( src )

# vi: ts=2 sw=2 et
