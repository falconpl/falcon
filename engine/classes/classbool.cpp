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

#undef SRC
#define SRC "engine/classes/classbool.cpp"

#include <falcon/classes/classbool.h>
#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

namespace Falcon {

ClassBool::ClassBool() :
   Class( "Boolean", FLC_ITEM_BOOL )
   
{ 
}

ClassBool::~ClassBool()
{
}


void ClassBool::op_create( VMContext* ctx, int pcount ) const
{
   if ( pcount >= 1 )
   {
      Class* cls;
      void* inst;
      Item* itm = ctx->opcodeParams(pcount);
      if( itm->asClassInst( cls, inst ) )
      {
         // put the item in the stack, just in case.
         ctx->stackResult( pcount+1, *itm );
         ctx->pushCode( &m_OP_create_next );
         cls->op_isTrue( ctx, inst );
         // let the vm call our next
      }
      else
      {
         ctx->stackResult( pcount+1, Item( itm->isTrue() ) );
      }
   }
}


void ClassBool::NextOpCreate::apply_( const PStep*, VMContext* ctx )
{
   ctx->topData().setBoolean(ctx->topData().isTrue());
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


void ClassBool::store( VMContext*, DataWriter* dw, void* data ) const
{
   dw->write( static_cast<Item*>( data )->asBoolean() );
}


void ClassBool::restore( VMContext* , DataReader* dr, void* data ) const
{
   bool value;
   dr->read( value );
   static_cast<Item*>( data )->setBoolean(value);
}


void ClassBool::describe( void *instance, String& target, int, int ) const
{
   target += static_cast<Item*>( instance )->asBoolean() ? "true" : "false";
}

// ===========================================================================

void ClassBool::op_isTrue( VMContext* ctx, void* ) const
{
   Item* iself;
   OpToken token( ctx, iself );
   token.exit( iself->asBoolean() );
}

void ClassBool::op_toString( VMContext* ctx, void* ) const
{
   Item* iself;
   OpToken token( ctx, iself );
   String* s = new String( iself->asBoolean() ? "true" : "false" );
   token.exit( s );
}

}

/* end of classbool.cpp */
