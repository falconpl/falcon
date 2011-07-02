/*
   FALCON - The Falcon Programming Language.
   FILE: classbool.cpp

   Class handler managing Boolean items.
   -------------------------------------------------------------------
   Author: Francesco Magliocca
   Begin: Sun, 19 Jun 2011 12:40:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/classbool.h>
#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>
#include <falcon/optoken.h>

namespace Falcon {

ClassBool::ClassBool() :
   Class( "Boolean", FLC_ITEM_BOOL )
   
{ 
}

ClassBool::~ClassBool()
{
}


void ClassBool::op_create( VMachine* vm, int pcount ) const
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


void ClassBool::NextOpCreate::apply_( const PStep*, VMachine* vm )
{
   vm->currentContext()->topData().setBoolean(vm->currentContext()->topData().isTrue());
}

void ClassBool::dispose( void *self ) const
{

   Item *data = static_cast<Item*>( self );

   delete data;

}


void* ClassBool::clone( void *self ) const
{
   Item *result = new Item;

   *result = *static_cast<Item*>( self );

   return result;
}


void ClassBool::serialize( DataWriter*, void* ) const
{
   // TODO
}


void* ClassBool::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}

void ClassBool::describe( void *instance, String& target, int, int ) const
{
   target += static_cast<Item*>( instance )->asBoolean() ? "true" : "false";
}

// ===========================================================================

void ClassBool::op_isTrue( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   token.exit( iself->asBoolean() );
}

void ClassBool::op_toString( VMachine *vm, void* ) const
{
   Item* iself;
   OpToken token( vm, iself );
   String* s = new String( iself->asBoolean() ? "true" : "false" );
   token.exit( s );
}

}

/* end of classbool.cpp */
