/* config.h.cmake */

/* Define if you have the <curses.h> header file. */
#cmakedefine HAVE_CURSES_H

/* Define if you have getpwnam_r and getpwuid_r that are draft POSIX.1
   versions. */
#cmakedefine HAVE_GETPW_R_DRAFT

/* Define if you have getpwnam_r and getpwuid_r that are POSIX.1
   compatible. */
#cmakedefine HAVE_GETPW_R_POSIX

/* Define if you have the `issetugid' function. */
#cmakedefine HAVE_ISSETUGID

/* Define if you have the <ncurses.h> header file. */
#cmakedefine HAVE_NCURSES_H

/* Define if you have the `reallocarr' function. */
#cmakedefine HAVE_REALLOCARR

/* Define if you have the `secure_getenv' function. */
#cmakedefine HAVE_SECURE_GETENV

/* Define if you have the <stdint.h> header file. */
#cmakedefine HAVE_STDINT_H

/* Define if you have the `strlcat' function. */
#cmakedefine HAVE_STRLCAT

/* Define if you have the `strlcpy' function. */
#cmakedefine HAVE_STRLCPY

/* Define if you have the <sys/types.h> header file. */
#cmakedefine HAVE_SYS_TYPES_H

/* Define if you have the <termcap.h> header file. */
#cmakedefine HAVE_TERMCAP_H

/* Define if you have the <term.h> header file. */
#cmakedefine HAVE_TERM_H

/* Define if the system has the type `u_int32_t'. */
#cmakedefine HAVE_U_INT32_T

/* Define if you have the `vis' function. */
#cmakedefine HAVE_VIS

/* Define if you have the `wcsdup' function. */
#cmakedefine HAVE_WCSDUP

/* Define if you have the `__secure_getenv' function. */
#cmakedefine HAVE___SECURE_GETENV

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif

#include "sys.h"
#undef SCCSID
#undef LIBC_SCCS
#undef lint
