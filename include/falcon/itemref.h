/*
   FALCON - The Falcon Programming Language.
   FILE: itemref.h

   Map holding variables and associated data for global storage.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 08 Jan 2013 18:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef FALCON_ITEMREF_H_
#define FALCON_ITEMREF_H_

#include <falcon/setup.h>
#include <falcon/atomic.h>
#include <falcon/item.h>

namespace Falcon
{

/** Reference to a in item variable.
 *
 * This structure holds an item plus an atomic reference counter.
 */
class ItemRef
{
public:
   ItemRef() :
      m_count(1)
   {}

   ItemRef( const Item& source ):
      m_data(source),
      m_count(1)
   {}

   void incref() { atomicInc(m_count);  }
   void decref() { if( atomicDec(m_count) == 0 ) delete this; }

   const Item& get() const { return m_data; }
   Item& get() { return m_data; }


private:
   ~ItemRef() {}

   Item m_data;
   atomic_int m_count;

};

}

#endif /* ITEMREF_H_ */

/* end of itemref.h */
