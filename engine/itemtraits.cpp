/*
   FALCON - The Falcon Programming Language
   FILE: itemtraits.cpp
   $Id: itemtraits.cpp,v 1.5 2007/08/11 00:11:54 jonnymind Exp $

   Traits for putting items in generic containers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 29 2006
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

//==========================================================
// Item traits using VM
//

int VMItemTraits::compare( const void *first, const void *second ) const
{
   const Item *f = (Item  *) first;
   const Item *s = (Item  *) second;

   return m_vm->compareItems( *f, *s );
}

namespace traits {
	ItemTraits t_item;
}

}


/* end of itemtraits.cpp */
