/*
   FALCON - The Falcon Programming Language.
   FILE: classnil.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classnil.cpp"

#include <falcon/classes/classnil.h>
#include <falcon/itemid.h>
#include <falcon/item.h>
#include <falcon/vmcontext.h>

namespace Falcon {

ClassNil::ClassNil():
   Class("Nil", FLC_ITEM_NIL )
{
}


ClassNil::~ClassNil()
{
}


void ClassNil::op_create( VMContext* ctx, int pcount ) const
{
   ctx->stackResult( pcount + 1, Item( ) );
}


void ClassNil::dispose( void* ) const
{
}


void* ClassNil::clone( void* ) const
{
   return 0;
}


void ClassNil::store( VMContext*, DataWriter* , void* ) const
{
   // Nothing to write.
}

void ClassNil::restore( VMContext* , DataReader* , void* ) const
{
   // nothing to read.
}

void ClassNil::describe( void*, String& target, int, int ) const
{
   target = "Nil";
}

//=======================================================================
//

void ClassNil::op_isTrue( VMContext* ctx, void* ) const
{
   ctx->stackResult( 1, false );
}
}

/* end of ClassNil.cpp */