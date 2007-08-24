/*
   FALCON - The Falcon Programming Language.
   FILE: flc_heap.h
   $Id: heap.h,v 1.1.1.1 2006/10/08 15:05:38 gian Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio set 30 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Short description
*/

#ifndef flc_flc_heap_H
#define flc_flc_heap_H

#ifdef HAVE_CONFIG_H
   #include <config.h>
#endif

#ifdef WIN32
   #include <falcon/heap_win.h>
#else
   #include <sys/mman.h>

   #if !defined( MAP_ANONYMOUS )
      /* Make compatible with mac OSX and BSD */
      #if defined(MAP_ANON)
         #define MAP_ANONYMOUS      MAP_ANON
         #include <falcon/heap_linux.h>
      #else
         #include <falcon/heap_unix.h>
      #endif
   #else
      #include <falcon/heap_linux.h>
   #endif
#endif

#endif

/* end of flc_heap.h */
