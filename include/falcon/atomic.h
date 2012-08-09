/*
   FALCON - The Falcon Programming Language.
   FILE: atomic.h

   Multithreaded extensions -- atomic operations.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 09 Aug 2012 00:01:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ATOMIC_H
#define FALCON_ATOMIC_H

#include <falcon/setup.h>
#include <falcon/types.h>

#ifdef FALCON_SYSTEM_WIN
#include <falcon/atomic_win.h>
#else
// TODO: atomic on other platforms
#include <falcon/atomic_gcc.h>
#endif

#endif

/* end of atomic.h */
