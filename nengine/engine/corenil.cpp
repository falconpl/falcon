/*
   FALCON - The Falcon Programming Language.
   FILE: corenil.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/corenil.h>
#include <falcon/itemid.h>
#include <falcon/item.h>

#include "falcon/vm.h"

namespace Falcon {

CoreNil::CoreNil():
   Class("Nil", FLC_ITEM_NIL )
{
}


CoreNil::~CoreNil()
{
}


void CoreNil::op_create( VMachine* vm, int pcount ) const
{
   vm->stackResult( pcount + 1, Item( ) );
}


void CoreNil::dispose( void* ) const
{
}


void* CoreNil::clone( void* ) const
{
   return 0;
}


void CoreNil::serialize( DataWriter*, void* ) const
{
}


void* CoreNil::deserialize( DataReader* ) const
{
   return 0;
}

void CoreNil::describe( void*, String& target, int, int ) const
{
   target = "Nil";
}

//=======================================================================
//

void CoreNil::op_isTrue( VMachine *vm, void* ) const
{
   vm->stackResult( 1, false );
}
}

/* end of CoreNil.cpp */
