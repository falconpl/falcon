/*
   FALCON - The Falcon Programming Language.
   FILE: syncqueue.h

   Synchronous queue template class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 04 Nov 2012 18:32:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYNCQUEUE_H_
#define _FALCON_SYNCQUEUE_H_

#include <falcon/setup.h>

#if defined(FALCON_SYSTEM_UNIX) || defined(FALCON_SYSTEM_MAC)
#include <falcon/syncqueue_posix.h>
#else

/* Vista has condition variables */
#if WINVER >= 0x600
   #include <falcon/syncqueue_win6.h>
#else
   /* Fallback to less-performing event-based queues. */
   #include <falcon/syncqueue_win.h>
#endif

#endif

#endif

/* end of syncqueue.h */
