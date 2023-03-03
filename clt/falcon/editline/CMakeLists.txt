cmake_minimum_required(VERSION 3.16.3)
project(Editline)

find_package(Curses REQUIRED)
set(HAVE_CURSES_H ${CURSES_HAVE_CURSES_H})
set(HAVE_NCURSES_H ${CURSES_HAVE_NCURSES_H} )

add_subdirectory(src)
#add_subdirectory(examples)

include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckTypeSize)
include(CheckSymbolExists)
check_include_file("sys/cdefs.h" HAVE_SYS_CDEFS_H)
check_include_file("term.h" HAVE_TERM_H)
check_function_exists("getpwuid_r" HAVE_GETPW_R_POSIX)
check_type_size("u_int32_t" U_INT32_T)
check_symbol_exists(alloca alloca.h HAVE_ALLOCA)
set(HAVE_ALLOCA_H ${HAVE_ALLOCA})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)
              
