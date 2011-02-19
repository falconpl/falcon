/*
   FALCON - The Falcon Programming Language.
   FILE: coreint.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/coreint.h>
#include <falcon/itemid.h>
#include <falcon/item.h>

namespace Falcon {

CoreInt::CoreInt():
   Class("Integer", FLC_ITEM_INT )
{
}


CoreInt::~CoreInt()
{
}


void* CoreInt::create(void* creationParams ) const
{
   Item* ptr = new Item;
   *ptr = int64(creationParams);
   return ptr;
}


void CoreInt::dispose( void* self ) const
{
   Item* data = (Item*) self;
   delete data;
}


void* CoreInt::clone( void* source ) const
{
   Item* ptr = new Item;
   *ptr = *(Item*) source;
   return ptr;
}


void CoreInt::serialize( Stream* stream, void* self ) const
{
   //TODO
}


void* CoreInt::deserialize( Stream* stream ) const
{
   //TODO
}

void CoreInt::describe( void* instance, String& target ) const
{
   target.N(((Item*) instance)->asInteger() );
}

//=======================================================================
//

void CoreInt::op_isTrue( VMachine *vm, void* self, Item& target ) const
{
   target.setBoolean( ((int64)self) != 0 );
}
}

/* end of coreint.cpp */
