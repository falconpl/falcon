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

namespace Falcon {

ClassRange::ClassRange():
   Class("Range", FLC_CLASS_ID_RANGE )
{
}


ClassRange::~ClassRange()
{
}


//void ClassRange::dispose( void* self ) const
//{
   //ItemDict* f = static_cast<ItemDict*>(self);
   //delete f;
//}


void* ClassRange::clone( void* source ) const
{
   return Range(*static_cast<Range*>(source));
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

void ClassRange::describe( void* instance, String& target, int maxDepth, int maxLen ) const
{
   if( maxDepth == 0 )
   {
      target = "...";
      return;
   }

   //ItemDict* dict = static_cast<ItemDict*>(instance);
   //dict->describe(target, maxDepth, maxLen);
}



void ClassRange::gcMark( void* self, uint32 mark ) const
{
   Range& range = *static_cast<Range*>(self);
   range.gcMark( mark );
}

bool gcCheck( void* self, uint32 mark ) const
{
   Range& range = *static_cast<Range*>(self);
   if ( range.gcMark() < mark)
   {
      dispose(self);
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
}

void ClassRange::enumeratePV( void*, Class::PVEnumerator& ) const
{
   // EnumerateVP doesn't normally return static methods.
}

//=======================================================================
//
void ClassRange::op_create( VMContext* ctx, int pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   int64* inst = new int64[3];
   inst[0] = inst[1] = inst[2] = 0;
   ctx->stackResult( pcount + 1, FALCON_GC_STORE( coll, this, inst ) );
}

void ClassRange::op_add( VMContext*, void* ) const
{
   //TODO
}

void ClassRange::op_isTrue( VMContext* ctx, void* self ) const
{
   ctx->stackResult( 1, static_cast<ItemDict*>(self)->size() != 0 );
}

void ClassRange::op_toString( VMContext* ctx, void* self ) const
{
   String s;
   s.A("[Dictionary of ").N((int64)static_cast<ItemDict*>(self)->size()).A(" elements]");
   ctx->stackResult( 1, s );
}


void ClassRange::op_getProperty( VMContext* ctx, void* self, const String& property ) const
{
   Class::op_getProperty( ctx, self, property );
}

void ClassRange::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *index, *dict_item;
   ctx->operands( index, dict_item );

   ItemDict& dict = *static_cast<ItemDict*>(self);
   Item* result = dict.find( *index );

   if( result != 0 )
   {
      ctx->stackResult( 2, *result );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__, __FILE__ ) );
   }
}

void ClassRange::op_setIndex( VMContext* ctx, void* self ) const
{
   Item *value, *index, *dict_item;
   ctx->operands( value, index, dict_item );

   ItemDict& dict = *static_cast<ItemDict*>(self);
   dict.insert( *index, *value );
   ctx->stackResult(3, *value);
}


}

/* end of classdict.cpp */
