/*
   FALCON - The Falcon Programming Language.
   FILE: detmempool.h
   $Id: detmempool.h,v 1.1 2007/01/28 16:36:46 jonnymind Exp $

   Deterministic memory pool
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom gen 28 2007
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
   Deterministic memory pool declaration.
*/

#ifndef flc_detmempool_H
#define flc_detmempool_H

#include <falcon/setup.h>
#include <falcon/mempool.h>


namespace Falcon {

/** Deterministic Memory Pool.
   This class works exactly as a basic memory pool, with the exception
   that if a garbage collection loop does not terminate by a certain
   time, the loop is interrupted.

   This may result in none of the garbage to be reclaimed; so the applications
   usign this memory pool must at times disable the collection timeout,
   or set it to a sufficently wide interval, so that a full collection may
   take place. However, the application stays in control of the garbage
   collection process.

   Sophisticated applications may derive this class to provide automatic
   timing strategies that would turn temporarily off the timeout when
   allocated memory becomes too huge.
*/

class DetMemPool: public MemPool
{
   uint32 m_msTarget;
   bool gcDetMark();
   void gcDetSweep();

public:
   DetMemPool() {}
   virtual bool performGC();
};

}

#endif

/* end of detmempool.h */
