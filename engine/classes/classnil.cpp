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
   m_bIsFlatInstance = true;
}


ClassNil::~ClassNil()
{
}


bool ClassNil::op_init( VMContext*, void* instance, int ) const
{
   Item* item = static_cast<Item*>(instance);
   item->setNil();
   return false;
}


void ClassNil::dispose( void* ) const
{
}


void* ClassNil::clone( void* inst ) const
{
   return inst;
}

void* ClassNil::createInstance() const
{
   return 0;
}

void ClassNil::store( VMContext*, DataWriter* , void* ) const
{
   // Nothing to write.
}

void ClassNil::restore( VMContext* ctx, DataReader*) const
{
   // just to be sure.
   ctx->pushData( Item() );
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
