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

bool PrototypeClass::op_init( VMContext* ctx, void* instance, int32 params ) const
{
   FlexyDict *value = static_cast<FlexyDict*>(instance);
   value->setBaseType(true);

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

   return false;
}


void PrototypeClass::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   FlexyDict *self = static_cast<FlexyDict*>(instance);
   
   if( ! self->isBaseType() )
   {
      FlexyClass::op_call( ctx, paramCount, instance );
      return;
   }
   
   FlexyDict* child = new FlexyDict;
   child->base().resize(1);
   child->base()[0].setUser(this, self);  

   Item* iNewSelf = ctx->opcodeParams(paramCount + 1);
   iNewSelf->setUser(this, child);
   Item* init = self->find("_init");
   if( init != 0 && init->isMethod() )
   {
      ctx->callInternal( init->asMethodFunction(), paramCount, *iNewSelf );
   }
   else
   {
      ctx->popData( paramCount );
   }             
}


void PrototypeClass::op_getProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict* fd = static_cast<FlexyDict*>(self);

   // check the special property _base
   if( prop == "_base" )
   {
      ctx->topData() = Item(fd->base().handler(), &fd->base());
   }   
   else if( prop == "_prototype" )
   {
      if( fd->base().length() > 0 )
      {
         ctx->topData() = fd->base()[0];
      }
      else {
         ctx->topData().setNil();
      }
   }
   else if( prop == "_isBase" )
   {
      ctx->topData().setBoolean( fd->isBaseType() );
   }
   else if( prop == "_init" )
   {
      Item* init = fd->find("_init");
      if( init == 0 || ! init->isMethod() ) {
         ctx->topData().setNil();
      }
      else {
         ctx->topData() = *init;
      }
   }
   else 
   {
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
               Item& result = ctx->topData();
               if( result.isMethod() && result.asClass() == this )
               {
                  //result.content.data.ptr.pInst = self;
               }
               return;
            }
         }
      }
      
      // fall back to the starndard system
      Class::op_getProperty( ctx, self, prop );
   }
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
   else if( prop == "_prototype" )
   {
      ctx->popData();
      Item& value = ctx->topData();
      
      if( fd->base().length() < 1 )
      {
         fd->base().resize(1);
      }
      
      value.copied();
      fd->base()[0] = value;
      return;
   }
   else if( prop == "_isBase" )
   {
      ctx->popData();
      Item& value = ctx->topData();      
      fd->setBaseType( value.isTrue() );
      return;
   }

   FlexyClass::op_setProperty( ctx, self, prop );
}


void PrototypeClass::describe( void* instance, String& target, int depth, int maxlen) const
{
   String instName;
   
   FlexyDict* fd = static_cast<FlexyDict*>(instance);
   ItemArray& base = fd->base();
   if( base.length() > 0 )
   {
      Class* cls;
      void* inst;
      base[0].forceClassInst(cls, inst);
      if( cls == this )
      {
         Item* iname = static_cast<FlexyDict*>(inst)->find("_name");
         if( iname != 0 && iname->isString() )
         {
            instName = *iname->asString();
         }
         else {
            instName = name();
         }
      }
      else {
         instName = cls->name();
      }
   }
   else
   {
      instName = name();
   }
   
   target.size(0);
   target += instName;
   if( fd->isBaseType() ) {
      target += "*";
   }

   
   if( depth != 0 )
   {
      String tgt;
      fd->describe( tgt, depth, maxlen );
      target += "{" + tgt + "}";
   }
   else
   {
      target += "{...}";
   }  
}

}

/* end of prototypeclass.cpp */

