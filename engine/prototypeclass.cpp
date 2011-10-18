/*
   FALCON - The Falcon Programming Language.
   FILE: prototypeclass.cpp

   Class holding more user-type classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 06:35:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/prototypeclass.cpp"

#include <falcon/prototypeclass.h>
#include <falcon/flexydict.h>
#include <falcon/vmcontext.h>

#include <falcon/errors/accesserror.h>

#include <map>
#include <vector>
#include <cstring>

namespace Falcon
{


PrototypeClass::PrototypeClass():
   FlexyClass( "Prototype" )
  
{
}


PrototypeClass::~PrototypeClass()
{
}


bool PrototypeClass::hasProperty( void* self, const String& prop ) const
{
   FlexyDict* fd = static_cast<FlexyDict*>(self);
   if( fd->hasProperty( prop) )
   {
      return true;
   }

   // nope -- search across bases.
   for( length_t i = 0; i < fd->base().length(); ++i )
   {
      Class* cls;
      void* data;
      fd->base()[i].forceClassInst( cls, data );
      if( cls->hasProperty( data, prop ) )
      {
         return true;
      }
   }
   
   return false;
}

//=========================================================
// Operators.
//

void PrototypeClass::op_create( VMContext* ctx, int32 params ) const
{
   static Collector* coll = Engine::instance()->collector();
   static Class* cls =  Engine::instance()->protoClass();

   FlexyDict *value = new FlexyDict;

   // we must create the prototype with the given bases.
   if( params > 0 )
   {
      ItemArray& base = value->base();
      base.reserve( params );

      Item* result = ctx->opcodeParams(params);
      Item* end = result + params;
      while( result < end )
      {
         base.append( *result );
         ++result;
      }
   }

   ctx->stackResult( params+1, FALCON_GC_STORE( coll, cls, value ) );
}


void PrototypeClass::op_getProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict* fd = static_cast<FlexyDict*>(self);

   // check the special property _base
   if( prop == "_base" )
   {
      ctx->topData().setArray(&fd->base());
      return;
   }

   Item* item = fd->find( prop );
   if( item != 0 )
   {
      ctx->stackResult(1, *item);
      return;
   }
   else
   {
      // nope -- search across bases.
      const ItemArray& base = fd->base();
      for( length_t i = 0; i < base.length(); ++i )
      {
         Class* cls;
         void* data;
         base[i].forceClassInst( cls, data );
         if( cls->hasProperty( data, prop ) )
         {
            cls->op_getProperty( ctx, data, prop );
            return;
         }
      }
   }

   // fall back to the starndard system
   Class::op_getProperty( ctx, self, prop );
}


void PrototypeClass::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict* fd = static_cast<FlexyDict*>(self);

   // check the special property _base
   if( prop == "_base" )
   {
      ctx->popData();
      Item& value = ctx->topData();
      if( value.isArray() )
      {
         fd->base().resize(0);
         fd->base().merge( *value.asArray() );
      }
      else
      {
         fd->base().resize(1);
         value.copied();
         fd->base()[0] = value;
      }
      return;
   }

   FlexyClass::op_setProperty( ctx, self, prop );
}


}

/* end of prototypeclass.cpp */

