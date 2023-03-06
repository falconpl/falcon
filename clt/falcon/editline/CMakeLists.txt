#/# Editline for Falcon

cmake_minimum_required( VERSION 3.16.3 )
project( Editline )

message( STATUS "Configuring Editline ..." )

find_package( Curses REQUIRED )
set( HAVE_CURSES_H ${CURSES_HAVE_CURSES_H} )
set( HAVE_NCURSES_H ${CURSES_HAVE_NCURSES_H} )
set( HAVE_LIBCURSES ${CURSES_HAVE_CURSES_H} )
set( HAVE_LIBNCURSES ${CURSES_HAVE_NCURSES_H} )

include( CheckFunctionExists )
include( CheckIncludeFile )
include( CheckSourceCompiles )
include( CheckSymbolExists )
include( CheckTypeSize )

set( CMAKE_REQUIRED_DEFINITIONS
  -D_GNU_SOURCE # secure_getenv
  #-D_POSIX_C_SOURCE=200809L
  #-D_REGEX_RE_COMP # deprecated functions
  #-D_XOPEN_SOURCE=500
)

# misc includes
check_include_file( dlfcn.h HAVE_DLFCN_H )
check_include_file( fcntl.h HAVE_FCNTL_H )
check_include_file( inttypes.h HAVE_INTTYPES_H )
check_include_file( limits.h HAVE_LIMITS_H )
check_include_file( malloc.h HAVE_MALLOC_H )
check_include_file( minix/config.h HAVE_MINIX_CONFIG_H )
check_include_file( stdint.h HAVE_STDINT_H )
check_include_file( stdio.h HAVE_STDIO_H )
check_include_file( sys/cdefs.h HAVE_SYS_CDEFS_H )
check_include_file( sys/ioctl.h HAVE_SYS_IOCTL_H )
check_include_file( sys/param.h HAVE_SYS_PARAM_H )
check_include_file( sys/stat.h HAVE_SYS_STAT_H )
check_include_file( sys/types.h HAVE_SYS_TYPES_H )
check_include_file( sys/wait.h HAVE_SYS_WAIT_H )
check_include_file( term.h HAVE_TERM_H )

# ctype.h
check_include_file ( ctype.h HAVE_CTYPE_H )
if ( HAVE_CTYPE_H )
  check_symbol_exists( isascii ctype.h HAVE_ISASCII )
endif()

# dirent.h
check_include_file( dirent.h HAVE_DIRENT_H )
if ( HAVE_DIRENT_H )
  check_source_compiles( C
    "#include <dirent.h>
    int main(void) { DIR *dir; }"
    HAVE_DIRENT_DIR
  )
  if ( NOT HAVE_DIRENT_DIR )
    set( HAVE_DIRENT_H OFF )
  else()
    check_symbol_exists(
      _DIRENT_HAVE_D_NAMLEN dirent.h HAVE_STRUCT_DIRENT_D_NAMLEN
    )
    check_source_compiles( C
      "#include <dirent.h>
      int main(void) { DIR* dir; int i = closedir(dir); }"
      CLOSEDIR_INT
    )
    if ( NOT CLOSEDIR_INT )
      set( CLOSEDIR_VOID ON )
    endif()
  endif()
endif()

# ndir.h
check_include_file( ndir.h HAVE_NDIR_H )
if ( HAVE_NDIR_H )
  check_source_compiles( C
    "#include <ndir.h>
    int main(void) { DIR *dir; }"
    HAVE_NDIR_DIR
  )
  if ( NOT HAVE_NDIR_DIR )
    set( HAVE_NDIR_H OFF )
  endif()
endif()

# pwd.h
check_include_file( pwd.h HAVE_PWD_H )
if ( HAVE_PWD_H )
  check_symbol_exists( endpwent pwd.h HAVE_ENDPWENT )
  check_symbol_exists( getpwnam_r pwd.h HAVE_GETPWNAM_R_POSIX )
  check_symbol_exists( getpwuid_r pwd.h HAVE_GETPWUID_R_POSIX )
  if ( HAVE_GETPWNAM_R_POSIX AND HAVE_GETPWUID_R_POSIX )
    set( HAVE_GETPW_R_POSIX ON )
  endif()
endif()

# regex.h
check_include_file( regex.h HAVE_REGEX_H )
if ( HAVE_REGEX_H )
  check_symbol_exists( regcomp regex.h HAVE_REGCOMP )
  check_symbol_exists( re_comp regex.h HAVE_RE_COMP )
endif()

# stdlib.h
check_include_file( stdlib.h HAVE_STDLIB_H )
if ( HAVE_STDLIB_H )
  check_symbol_exists( reallocarr stdlib.h HAVE_REALLOCARR )
  check_symbol_exists( secure_getenv stdlib.h HAVE_SECURE_GETENV )
  check_symbol_exists( strtol stdlib.h HAVE_STRTOL )
endif()

# string.h
check_include_file( string.h HAVE_STRING_H )
if ( HAVE_STRING_H )
  check_symbol_exists( memchr string.h HAVE_MEMCHR )
  check_symbol_exists( memset string.h HAVE_MEMSET )
  check_symbol_exists( strchr string.h HAVE_STRCHR )
  check_symbol_exists( strcspn string.h HAVE_STRCSPN )
  check_symbol_exists( strdup string.h HAVE_STRDUP )
  check_symbol_exists( strerror string.h HAVE_STRERROR )
  check_symbol_exists( strlcat string.h HAVE_STRLCAT )
  check_symbol_exists( strlcpy string.h HAVE_STRLCPY )
  check_symbol_exists( strrchr string.h HAVE_STRRCHR )
  check_symbol_exists( strstr string.h HAVE_STRSTR )
endif()

# strings.h
check_include_file( strings.h HAVE_STRINGS_H )
if ( HAVE_STRINGS_H )
  check_symbol_exists( strcasecmp strings.h HAVE_STRCASECMP )
endif()

# sys/dir.h
check_include_file( sys/dir.h HAVE_SYS_DIR_H )
if ( HAVE_SYS_DIR_H )
  check_source_compiles( C
    "#include <sys/dir.h>
    int main(void) { DIR *dir; }"
    HAVE_SYS_DIR_DIR
  )
  if ( NOT HAVE_SYS_DIR_DIR )
    set( HAVE_SYS_DIR_H OFF )
  endif()
endif()

# sys/ndir.h
check_include_file( sys/ndir.h HAVE_SYS_NDIR_H )
if ( HAVE_SYS_NDIR_H )
  check_source_compiles( C
    "#include <sys/ndir.h>
    int main(void) { DIR *dir; }"
    HAVE_SYS_NDIR_DIR
  )
  if ( NOT HAVE_SYS_NDIR_DIR )
    set( HAVE_SYS_NDIR_H OFF )
  endif()
endif()

# termcap.h
check_include_file( termcap.h HAVE_TERMCAP_H )
if ( HAVE_TERMCAP_H )
  # assume the library exists...
  set( HAVE_LIBTERMCAP ON )
endif()

# unistd.h
check_include_file( unistd.h HAVE_UNISTD_H )
if ( HAVE_UNISTD_H )
  check_symbol_exists( issetugid unistd.h HAVE_ISSETUGID )
  check_symbol_exists( fork unistd.h HAVE_FORK )
  # if it compiles, let's assume it's correctly implemented...
  if ( HAVE_FORK )
    check_source_compiles( C
      "#include <unistd.h>
      int main(void) { pid_t pid = fork(); }"
      HAVE_WORKING_FORK
    )
  endif()
endif()

# vfork.h & vfork
# unused in editline but for the sake of completeness
check_include_file( vfork.h HAVE_VFORK_H )
check_symbol_exists( vfork "unistd.h;vfork.h" HAVE_VFORK )
if ( HAVE_VFORK )
  # if it compiles, let's assume it's correctly implemented...
  if ( HAVE_VFORK_H )
    check_source_compiles( C
      "#include <vfork.h>
      int main(void) { pid_t pid = vfork(); }"
      HAVE_WORKING_VFORK
    )
  else()
    check_source_compiles( C
      "#include <unistd.h>
      int main(void) { pid_t pid = vfork(); }"
      HAVE_WORKING_VFORK
    )
  endif()
endif()

# vis.h
check_include_file( vis.h HAVE_VIS_H )
if ( HAVE_VIS_H )
  check_symbol_exists( strunvis vis.h HAVE_STRUNVIS )
  check_symbol_exists( strvis vis.h HAVE_STRVIS )
  check_symbol_exists( vis vis.h HAVE_VIS )
endif()

# wchar.h
check_include_file( wchar.h HAVE_WCHAR_H )
if ( HAVE_WCHAR_H )
  check_symbol_exists( wcsdup wchar.h HAVE_WCSDUP )
endif()


# misc type checks
check_type_size( u_int32_t U_INT32_T )


# libtinfo is libncurses
set( HAVE_LIBTINFO OFF )

# where is?
#check_symbol_exists(__secure_getenv XXX HAVE___SECURE_GETENV)

# lstat is unused anyway
set( LSTAT_FOLLOWS_SLASHED_SYMLINK ON )


configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

add_subdirectory( src )

# vi: ts=2 sw=2 et
