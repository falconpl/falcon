/*
   FALCON - The Falcon Programming Language.
   FILE: classmethod.cpp

   Method object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 15:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classmethod.cpp"

#include <falcon/classes/classmethod.h>
#include <falcon/vmcontext.h>
#include <falcon/function.h>


#include <falcon/errors/paramerror.h>

namespace Falcon
{

ClassMethod::ClassMethod():
   Class("Method", FLC_ITEM_METHOD )
{
   m_bIsFlatInstance = true;
}


ClassMethod::~ClassMethod()
{}


void ClassMethod::dispose( void* ) const
{
}


void* ClassMethod::clone( void* self ) const
{   
   return self;
}

void* ClassMethod::createInstance() const
{
   // this is a flat class
   return 0;
}

void ClassMethod::serialize( DataWriter*, void* ) const
{
   //TODO
}


void* ClassMethod::deserialize( DataReader* ) const
{
   //TODO
   return 0;
}


void ClassMethod::describe( void* instance, String& target, int depth, int maxLen ) const
{
   Item* item = (Item*) instance;
   target = item->describe( depth, maxLen );
}


void ClassMethod::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   cb( "origin" );
   cb( "source" );
}


void ClassMethod::enumeratePV( void* self, PVEnumerator& cb ) const
{
   Item& src = *(Item*) self;
   Item copy = *(Item*) self;

   copy.unmethodize();
   Item func = src.asMethodFunction();

   cb( "origin", copy );
   cb( "source", func );
}


bool ClassMethod::hasProperty( void*, const String& prop ) const
{
   return
            prop == "origin"
        ||  prop == "source"
    ;
}

//=============================================================
bool ClassMethod::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   Item* item = static_cast<Item*>(instance);
   
   if( pcount == 2 )
   {
      Item& src = ctx->opcodeParam(1);
      Item& func = ctx->opcodeParam(0);
      if( func.isFunction() )
      {
         *item = src;
         item->methodize( func.asFunction() );
         return false;
      }
   }

   throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
      .origin( ErrorParam::e_orig_vm )
      .extra("X,F") );
}


void ClassMethod::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   Item copy = *static_cast<Item*>(self);
   Function* fmth = copy.asMethodFunction();
   copy.unmethodize();
   ctx->callInternal( fmth, paramCount, copy );
}


void ClassMethod::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   Item* self = static_cast<Item*>(instance);

   if( prop == "origin" )
   {
      Item copy = *self;
      copy.unmethodize();
      ctx->stackResult(1, copy );
   }
   else if( prop == "source" )
   {
      Function* f = self->asMethodFunction();
      if( f->isCompatibleWith( Function::e_c_synfunction ) )
      {
         ctx->stackResult(1, f );
      }
      else {
         // we cannot un-methodize C functions.
         ctx->stackResult(1, Item() );
      }
   }
   else {
      Class::op_getProperty(ctx, instance, prop );
   }
}

}

/* end of classmethod.cpp */
