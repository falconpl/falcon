/*
   FALCON - The Falcon Programming Language.
   FILE: shared_private.h

   VM Scheduler managing waits and sleeps of contexts -- private part
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jul 2012 16:52:49 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SHARED_PRIVATE_H_
#define _FALCON_SHARED_PRIVATE_H_

#include <falcon/shared.h>
#include <deque>

namespace Falcon {

class Shared::Private {
public:
   Private::Private( int32 signals = 0 ):
      m_signals(signals)
   {}

   typedef std::deque<VMContext*> ContextList;

   Mutex m_mtx;
   ContextList m_waiters;

   int m_signals;
};

}

/* end of shared_private.h */
