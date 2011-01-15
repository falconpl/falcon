/*
   FALCON - The Falcon Programming Language.
   FILE: referenceable.h

   Reference count system.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 09 Jan 2011 13:37:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_REFERENCEABLE_H
#define FALCON_REFERENCEABLE_H

#include <falcon/mt.h>

namespace Falcon
{

/** Base interface for reference based objects.
 *
 * This class, or better, this interface can be used to add
 * a reference count to objects that may be multi-referenced.
 *
 * This may happens to objects shared by multiple falcon core object instances,
 * or objects shared between a Falcon object and an application. In that case,
 * the class may implement this class as a simple mean to keep track of multiple
 * subjects interested in the objects.
 *
 * Falcon doesn't use reference count in its garbage collecting process, but
 * when an object may be assigned to the garbage collector AND to an external
 * entity that may transcend the GC lifetime, then it is necessary to cross
 * reference it.
 */
class FALCON_DYN_CLASS Referenceable
{
public:
   Referenceable():
      m_count(0)
   {}

   void incref() { atomicInc(m_count); }
   void decref() { if ( atomicDec(m_count) <= 0 ) delete this ; }

private:
   int m_count;
};

}

#endif
/* end of referenceable.h */
