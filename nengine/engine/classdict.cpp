/*
   FALCON - The Falcon Programming Language.
   FILE: classdict.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 15:33:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classdict.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/itemdict.h>


#include "falcon/accesserror.h"

namespace Falcon {

ClassDict::ClassDict():
   Class("Dictionary", FLC_CLASS_ID_DICT )
{
}


ClassDict::~ClassDict()
{
}


void ClassDict::dispose( void* self ) const
{
   ItemDict* f = static_cast<ItemDict*>(self);
   delete f;
}


void* ClassDict::clone( void* source ) const
{
   return static_cast<ItemDict*>(source)->clone();
}


void ClassDict::serialize( DataWriter*, void* ) const
{
   // todo
}


void* ClassDict::deserialize( DataReader* ) const
{
   //todo
   return 0;
}

void ClassDict::describe( void* instance, String& target, int maxDepth, int maxLen ) const
{
   if( maxDepth == 0 )
   {
      target = "...";
      return;
   }

   ItemDict* dict = static_cast<ItemDict*>(instance);
   dict->describe(target, maxDepth, maxLen);
}



void ClassDict::gcMark( void* self, uint32 mark ) const
{
   ItemDict& dict = *static_cast<ItemDict*>(self);
   dict.gcMark( mark );
}

void ClassDict::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   cb("len", true);
}

//=======================================================================
//
void ClassDict::op_create( VMContext* ctx, int pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   ctx->stackResult( pcount + 1, FALCON_GC_STORE( coll, this, new ItemDict ) );
}

void ClassDict::op_add( VMContext*, void* ) const
{
   //TODO
}

void ClassDict::op_isTrue( VMContext* ctx, void* self ) const
{
   ctx->stackResult( 1, static_cast<ItemDict*>(self)->size() != 0 );
}

void ClassDict::op_toString( VMContext* ctx, void* self ) const
{
   String s;
   s.A("[Dictionary of ").N((int64)static_cast<ItemDict*>(self)->size()).A(" elements]");
   ctx->stackResult( 1, s );
}


void ClassDict::op_getProperty( VMContext* ctx, void* self, const String& property ) const
{
   Class::op_getProperty( ctx, self, property );
}

void ClassDict::op_getIndex( VMContext* ctx, void* self ) const
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

void ClassDict::op_setIndex( VMContext* ctx, void* self ) const
{
   Item *value, *index, *dict_item;
   ctx->operands( value, index, dict_item );

   ItemDict& dict = *static_cast<ItemDict*>(self);
   dict.insert( *index, *value );
   ctx->stackResult(3, *value);
}


}

/* end of classdict.cpp */
