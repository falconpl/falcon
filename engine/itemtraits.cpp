/*
   FALCON - The Falcon Programming Language
   FILE: itemtraits.cpp

   Traits for putting items in generic containers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 29 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Traits for putting items in generic containers
*/

#include <falcon/itemtraits.h>
#include <falcon/vm.h>

namespace Falcon {

uint32 ItemTraits::memSize() const
{
   return sizeof( Item );
}

void ItemTraits::init( void *itemZone ) const
{
   Item *itm = (Item  *) itemZone;
   itm->setNil();
}

void ItemTraits::copy( void *targetZone, const void *sourceZone ) const
{
   Item *target = (Item  *) targetZone;
   const Item *source = (Item  *) sourceZone;

   target->copy( *source );
}

int ItemTraits::compare( const void *first, const void *second ) const
{
   const Item *f = (Item  *) first;
   const Item *s = (Item  *) second;
   return f->compare( *s );
}

void ItemTraits::destroy( void *item ) const
{
   // do nothing
}

bool ItemTraits::owning() const
{
   return false;
}

namespace traits {
	ItemTraits &t_item() { static ItemTraits *dt = new ItemTraits(); return *dt; }
}

}


/* end of itemtraits.cpp */
