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
   m_bIsFlatInstance = true;
}

ClassBool::~ClassBool()
{
}


bool ClassBool::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   Item* item = static_cast<Item*>(instance);
   
   // this will tell us if the instance comes from the stack.
   bool isInStack = instance == ctx->opcodeParam(pcount+1);
   if ( pcount >= 1 )
   {
      Class* cls;
      void* inst;
      Item* param = ctx->opcodeParams(pcount);
      
      if( param->asClassInst( cls, inst ) )
      {
         // put the item in the stack, just in case.
         ctx->pushCode( &m_OP_create_next );
         ctx->currentCode()->m_seqId = pcount;
         if ( isInStack ) ctx->currentCode()->m_seqId |=0x80000000;
         
         long depth = ctx->codeDepth();
         // we're in charge.
         if( ! isInStack ) {
            ctx->pushData( Item( this, instance ) );
         }
         ctx->pushData( Item( cls, inst ) );
         cls->op_isTrue( ctx, inst );
         if( ctx->codeDepth() != depth )
         {
            return true;
         }
         
         // we can progress right here.
         ctx->popCode();
         if( ! isInStack ) {
            item->setBoolean( ctx->topData().isTrue() );
            ctx->popData(2);
         }
         else {
            bool isTrue = ctx->topData().isTrue();
            ctx->popData();
            ctx->opcodeParam(pcount+1)->setBoolean(isTrue);
         }
      }
      else
      {
         item->setBoolean( param->isTrue() );
      }
   }
   else {
      item->setBoolean(false);
   }
   
   return false;
}


void ClassBool::NextOpCreate::apply_( const PStep*, VMContext* ctx )
{
   bool tof = ctx->topData().isTrue();
   int seqId = ctx->currentCode().m_seqId & 0xFFFFFFF;
   bool isInStack = (ctx->currentCode().m_seqId & 0x80000000) != 0;
   ctx->popCode(); // remove us
   
   if( isInStack )
   {
      ctx->popData( seqId );  // remove the parameters
      
      // store the value on top of the stack.
      // -- note, deep-flat instances necessarily need this, the pointer
      // to the instance might be screwed.
      ctx->topData().setBoolean( tof );
   }
   else {
      ctx->popCode(); // remove us
      ctx->popData(); // remove the called entity
      fassert( ctx->topData().asClass() == this );      
      static_cast<Item*>(ctx->topData().asInst())->setBoolean( tof );      
      ctx->popData(seqId + 1); // remove the params + the pointer to inst.
   }
}

void ClassBool::dispose( void* ) const
{
}

void* ClassBool::createInstance() const
{
   // this is a flat class.
   return 0;
}

void* ClassBool::clone( void *self ) const
{
   return self;
}


void ClassBool::store( VMContext*, DataWriter* dw, void* data ) const
{
   dw->write( static_cast<Item*>( data )->asBoolean() );
}


void ClassBool::restore( VMContext* , DataReader* dr, void*& data ) const
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
