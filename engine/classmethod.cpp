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
#define SRC "engine/classmethod.cpp"

#include <falcon/classmethod.h>
#include <falcon/paramerror.h>
#include <falcon/vmcontext.h>

namespace Falcon
{

ClassMethod::ClassMethod():
   Class("Method", FLC_ITEM_METHOD )
{}


ClassMethod::~ClassMethod()
{}


void ClassMethod::dispose( void* self ) const
{
   delete (Item*) self;
}


void* ClassMethod::clone( void* self ) const
{
   Item* ptr = new Item;
   *ptr = *(Item*) self;
   return ptr;
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
   cb( "origin", true );
}


void ClassMethod::enumeratePV( void* self, PVEnumerator& cb ) const
{
   Item copy = *(Item*) self;
   copy.unmethodize();
   cb( "origin", copy );   
}


bool ClassMethod::hasProperty( void*, const String& prop ) const
{
   if( prop == "origin" ) return true;
   
   return false;
}

//=============================================================
void ClassMethod::op_create( VMContext* ctx, int32 pcount ) const
{
   if( pcount == 2 )
   {
      Item& src = ctx->opcodeParam(1);
      Item& func = ctx->opcodeParam(0);
      if( func.isFunction() )
      {
         Item mth = src;
         mth.methodize( func.asFunction() );
         ctx->stackResult( pcount + 1, mth );
         return;
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
   ctx->call( fmth, paramCount, copy );
}

}

/* end of classmethod.cpp */
