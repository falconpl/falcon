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


void* CoreNil::create(void* creationParams ) const
{
   return 0;
}


void CoreNil::dispose( void* ) const
{
}


void* CoreNil::clone( void* source ) const
{
   return 0;
}


void CoreNil::serialize( DataWriter* stream, void* self ) const
{
}


void* CoreNil::deserialize( DataReader* stream ) const
{
   return 0;
}

void CoreNil::describe( void* instance, String& target ) const
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
