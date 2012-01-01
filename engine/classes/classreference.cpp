/*
   FALCON - The Falcon Programming Language.
   FILE: classreference.cpp

   Reference to remote items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Jul 2011 10:53:54 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classreference.cpp"

#include <falcon/fassert.h>
#include <falcon/classes/classreference.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/errors/paramerror.h>
#include <falcon/itemreference.h>

#include "falcon/itemarray.h"

namespace Falcon
{

ClassReference::ClassReference():
   Class("Reference", FLC_ITEM_REF)
{   
}


ClassReference::~ClassReference()
{  
}

Class* ClassReference::getParent( const String& ) const
{
   return 0;
}

void ClassReference::dispose( void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   // TODO: pool references
   delete ref;
}


void* ClassReference::clone( void* source ) const
{
   ItemReference* ref = static_cast<ItemReference*>(source);
   // TODO: pool references
   return new ItemReference(*ref);   
}

void ClassReference::store( VMContext*, DataWriter*, void* ) const {}
void ClassReference::restore( VMContext*, DataReader*, void*& empty ) const 
{
   empty = new ItemReference;
}

void ClassReference::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ItemReference* ref = static_cast<ItemReference*>(instance);
   subItems.append(ref->item());
}

void ClassReference::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ItemReference* ref = static_cast<ItemReference*>(instance);
   ref->item() = subItems[0];
}

void ClassReference::gcMark( void* self, uint32 mark ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   ref->gcMark(mark);
}


bool ClassReference::gcCheck( void* self, uint32 mark ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   return ref->gcMark() >= mark;
}


void ClassReference::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);

   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->enumerateProperties( data, cb );
}


void ClassReference::enumeratePV( void* self, PVEnumerator& cb ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->enumeratePV( data, cb );
}


bool ClassReference::hasProperty( void* self, const String& prop ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   return cls->hasProperty( data, prop );
}


void ClassReference::describe( void* self, String& target, int depth, int maxlen ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );

   String temp;
   cls->describe( data, temp, depth, maxlen );
   target = "Ref{" + temp + "}";
}


   
void ClassReference::op_create( VMContext* ctx, int32 pcount ) const
{
   if( pcount == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin(ErrorParam::e_orig_vm)
         .extra( "X") );
   }
   
   if( pcount > 1 )
   {
      ctx->popData(pcount-1);
   }
   
   if( ! ctx->topData().isReference() )
   {
      ItemReference::create(ctx->topData());
   }
   // topdata has been already turned into a reference by now.
}


void ClassReference::op_neg( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_neg( ctx, data );
}


void ClassReference::op_add( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_add( ctx, data );   
}


void ClassReference::op_sub( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_sub( ctx, data );      
}


void ClassReference::op_mul( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_mul( ctx, data );   
}


void ClassReference::op_div( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_div( ctx, data );   
}


void ClassReference::op_mod( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_mod( ctx, data );   
}


void ClassReference::op_pow( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_pow( ctx, data );   
}


void ClassReference::op_shr( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_shr( ctx, data );   
}


void ClassReference::op_shl( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_shl( ctx, data );   
}


void ClassReference::op_aadd( VMContext* ctx, void* self) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_aadd( ctx, data );   
}


void ClassReference::op_asub( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_asub( ctx, data );   
}


void ClassReference::op_amul( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_amul( ctx, data );   
}


void ClassReference::op_adiv( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_adiv( ctx, data );   
}


void ClassReference::op_amod( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_amod( ctx, data );   
}


void ClassReference::op_apow( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_apow( ctx, data );   
}

void ClassReference::op_ashr( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_ashr( ctx, data );   
}

void ClassReference::op_ashl( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_ashl( ctx, data );   
}

void ClassReference::op_inc( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_inc( ctx, data );   
}


void ClassReference::op_dec(VMContext* ctx, void* self) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_dec( ctx, data );   
}


void ClassReference::op_incpost(VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_incpost( ctx, data );   
}


void ClassReference::op_decpost(VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_decpost( ctx, data );   
}


void ClassReference::op_getIndex(VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_getIndex( ctx, data );   
}


void ClassReference::op_setIndex(VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_setIndex( ctx, data );   
}


void ClassReference::op_getProperty( VMContext* ctx, void* self, const String& prop) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_getProperty( ctx, data, prop );   
}


void ClassReference::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_setProperty( ctx, data, prop );   
}


void ClassReference::op_compare( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_compare( ctx, data );   
}


void ClassReference::op_isTrue( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_isTrue( ctx, data );   
}


void ClassReference::op_in( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_in( ctx, data );   
}


void ClassReference::op_provides( VMContext* ctx, void* self, const String& property ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_provides( ctx, data, property );   
}


void ClassReference::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_call( ctx, paramCount, data );   
}


void ClassReference::op_toString( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_toString( ctx, data );   
}


void ClassReference::op_iter( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_iter( ctx, data );   
}


void ClassReference::op_next( VMContext* ctx, void* self ) const
{
   ItemReference* ref = static_cast<ItemReference*>(self);
   Class* cls;
   void* data;
   ref->item().forceClassInst( cls, data );
   cls->op_next( ctx, data );   
}


}

/* end of classreference.cpp */
