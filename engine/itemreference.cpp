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

ItemReference* ItemReference::create( Item& source )
{
   static Class* cls = Engine::instance()->referenceClass();
   static Collector* coll = Engine::instance()->collector();

   if( source.isReference() )
   {
      return static_cast<ItemReference*>(source.asInst());
   }
   else
   {
      ItemReference* nr = new ItemReference( source );
      // generate a GC token for tne newly created entity.
      source.setUser( FALCON_GC_STORE( coll, cls, nr ) );
      // save the item as a reference.
      source.content.mth.ref = &nr->item();
      source.type(FLC_ITEM_REF);
      return nr;
   }
}

ItemReference* ItemReference::create( Item& source, Item &target )
{
   static Class* cls = Engine::instance()->referenceClass();
   static Collector* coll = Engine::instance()->collector();

   if( ! source.isReference() )
   {
      ItemReference* nr = new ItemReference( source );
      // save the item as a reference.
      source.setUser( FALCON_GC_STORE( coll, cls, nr ) );
      // save the item as a reference.
      source.content.mth.ref = &nr->item();
      source.type(FLC_ITEM_REF);
   }
   
   target = source;
   return static_cast<ItemReference*>(source.asInst());
}


bool ItemReference::referenceItem( Item& source )
{
   static Class* cls = Engine::instance()->referenceClass();
   static Collector* coll = Engine::instance()->collector();
   
   if( source.isReference() )
   {
      return false;
   }
   
   
   m_item = source;
   // generate a GC token for tne newly created entity.
   source.setUser( FALCON_GC_STORE( coll, cls, this ) );
   // save the item as a reference.
   source.content.mth.ref = &m_item;
   source.type(FLC_ITEM_REF);
   return true;
}

}
