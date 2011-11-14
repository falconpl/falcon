/*
   FALCON - The Falcon Programming Language.
   FILE: itemreference.cpp

   Class implementing a reference to an item.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 03 Nov 2011 19:13:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/itemreference.h>
#include <falcon/engine.h>
#include <falcon/classes/classreference.h>

namespace Falcon {

void ItemReference::reference( Item& target )
{
   static Class* cls = Engine::instance()->referenceClass();
   static Collector* coll = Engine::instance()->collector();

   if( target.isReference() )
   {
      m_item = *target.asReference();
   }
   else
   {
      m_item  = target;
      target.setUser( FALCON_GC_STORE( coll, cls, this) );
      target.content.mth.ref = &m_item;
      target.type( FLC_ITEM_REF );
   }
}

}
