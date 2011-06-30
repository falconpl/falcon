/*
   FALCON - The Falcon Programming Language.
   FILE: corebool.cpp

   Class defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Francesco Magliocca
   Begin: Sun, 19 Jun 2011 12:40:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/corebool.h>
#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>
#include <falcon/optoken.h>

namespace Falcon {

CoreBool::CoreBool() :
   Class( "Boolean", FLC_ITEM_BOOL )
   
{ 
}

CoreBool::~CoreBool()
{
}


void CoreBool::op_create( VMachine* vm, int pcount ) const
{
   if ( pcount >= 1 )
   {
      Class* cls;
      void* inst;
      Item* itm = vm->currentContext()->opcodeParams(pcount);
      if( itm->asClassInst( cls, inst ) )
      {
         // put the item in the stack, just in case.
         vm->stackResult( pcount+1, *itm );
         cls->op_isTrue( vm, inst );
         if( vm->wentDeep() )
         {
            return;
         }
         // if the item is not deep, then isTrue has already done what we want.
         // but better be sure
         vm->currentContext()->topData().setBoolean(vm->currentContext()->topData().isTrue());
      }
      else
      {
         vm->stackResult( pcount+1, Item( itm->isTrue() ) );
      }
   }
}


void CoreBool::NextOpCreate::apply_( const PStep*, VMachine* vm )
{
   vm->currentContext()->topData().setBoolean(vm->currentContext()->topData().isTrue());
}

void CoreBool::dispose( void *self ) const
{

   Item *data = static_cast<Item*>( self );

   delete data;

}


void* CoreBool::clone( void *self ) const
{
   Item *result = new Item;

   *result = *static_cast<Item*>( self );

   return result;
}


void CoreBool::serialize( DataWriter*, void* ) const
{
   // TODO
}


void* CoreBool::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}

void CoreBool::describe( void *instance, String& target, int, int ) const
{
   target += static_cast<Item*>( instance )->asBoolean() ? "true" : "false";
}

// ===========================================================================

void CoreBool::op_isTrue( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   token.exit( iself->asBoolean() );
}

void CoreBool::op_toString( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   String s;
   token.exit( s.A( iself->asBoolean() ? "true" : "false" ) );
}


}