/*
   FALCON - The Falcon Programming Language.
   FILE: classrange.h

   Standard language range object handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai & Paul Davey
   Begin: Mon, 25 Jul 2011 23:04 +1200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classrange.h>
#include <falcon/range.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>


#include "falcon/accesserror.h"
#include "falcon/accesstypeerror.h"

namespace Falcon {

ClassRange::ClassRange():
   Class("Range", FLC_CLASS_ID_RANGE )
{
}


ClassRange::~ClassRange()
{
}


void ClassRange::dispose( void* self ) const
{
   Range* f = static_cast<Range*>(self);   
   delete f;
}


void* ClassRange::clone( void* source ) const
{
   return new Range(*static_cast<Range*>(source));
}


void ClassRange::serialize( DataWriter*, void* ) const
{
   // todo
}


void* ClassRange::deserialize( DataReader* ) const
{
   //todo
   return 0;
}

void ClassRange::describe( void* instance, String& target, int maxDepth, int ) const
{
   if( maxDepth == 0 )
   {
      target = "...";
      return;
   }

   Range* r = static_cast<Range*>(instance);
   r->describe(target);
}



void ClassRange::gcMark( void* self, uint32 mark ) const
{
   Range& range = *static_cast<Range*>(self);
   range.gcMark( mark );
}


bool ClassRange::gcCheck( void* self, uint32 mark ) const
{
   Range& range = *static_cast<Range*>(self);
   if ( range.gcMark() < mark)
   {
      return false;
   }
   else
   {
      return true;
   }
}

void ClassRange::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   cb("len", true);
   cb("len_", true);
}

void ClassRange::enumeratePV( void*, Class::PVEnumerator& rator ) const
{
   Item temp;
   
   temp = 3;
   rator("len_", temp );
}


//=======================================================================
//
void ClassRange::op_create( VMContext* ctx, int pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   
   Range* inst = new Range;
   
   
   switch( pcount )
   {
      case 0:
         inst->start(0); 
         inst->setOpen(true);
         break;
         
      case 1: 
         inst->start(ctx->opcodeParam(0).forceInteger()); 
         inst->setOpen(true);
         break;
         
      case 2:
         inst->start(ctx->opcodeParam(1).forceInteger()); 
         if( ctx->opcodeParam(0).isNil() )
         {
            inst->setOpen(true);         
         }
         else
         {
            inst->end(ctx->opcodeParam(0).forceInteger());
         }
         break;
         
      default: // 3 or more.
         inst->start(ctx->opcodeParam(2).forceInteger()); 
         if( ctx->opcodeParam(1).isNil() )
         {
            inst->setOpen(true);         
         }
         else
         {
            inst->end(ctx->opcodeParam(1).forceInteger());
         }
         inst->step(ctx->opcodeParam(0).forceInteger()); 
         inst->setOpen(false);
         break;
   }
   
   ctx->stackResult( pcount + 1, FALCON_GC_STORE( coll, this, inst ) );
}


void ClassRange::op_getProperty( VMContext* ctx, void* self, const String& property ) const
{
   if( property == "len_" )
   {
      ctx->stackResult(1, 3);
   }
   
   Class::op_getProperty( ctx, self, property );
}

void ClassRange::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *item, *index;
   ctx->operands( item, index );
   Range& range = *static_cast<Range*>(self);

   if( index->isOrdinal() )
   {
      int64 v = index->forceInteger();
      if ( v < 0 ) v = 2+v;
      switch( v )
      {
         case 0:
            ctx->stackResult( 2, range.start() );
            break;
            
         case 1:
            ctx->stackResult( 2, range.isOpen()? Item() : Item(range.end()) );
            break;
            
         case 2:
            ctx->stackResult( 2, range.step() );
            break;
            
         default:
            throw new AccessError( ErrorParam(e_arracc, __LINE__, SRC )
               .origin(ErrorParam::e_orig_vm) );
      }      
   }
   else
   {
      throw new AccessTypeError( ErrorParam(e_param_type, __LINE__, SRC )
            .origin(ErrorParam::e_orig_vm) );
   }
}

void ClassRange::op_setIndex( VMContext* ctx, void* self ) const
{
   Item* value, *item, *index;
   ctx->operands( value, item, index );
   
   Range* range = static_cast<Range*>(self);

   if( index->isOrdinal() )
   {
      int64 v = index->forceInteger();
      if ( v < 0 ) v = 2+v;
      
      switch( v )
      {
         case 0:
            range->start( value->forceInteger() );
            ctx->popData(2);
            break;
            
         case 1: 
            if( value->isNil() )
            {
               range->setOpen(true);
            }
            else if ( value->isOrdinal() )
            {
               range->setOpen(false);                              
               range->end( value->forceInteger() );
            }
            
            ctx->popData(2);
            break;

         case 2:
            range->step( value->forceInteger() );
            ctx->popData(2);
            break;            
          
         default:
            throw new AccessError( ErrorParam(e_arracc, __LINE__, SRC )
               .origin(ErrorParam::e_orig_vm) );
      }      
   }
   else
   {
      throw new AccessTypeError( ErrorParam(e_param_type, __LINE__, SRC )
            .origin(ErrorParam::e_orig_vm) );
   }
}


}

/* end of classdict.cpp */
