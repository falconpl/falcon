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
#define SRC "engine/classreference.cpp"

#include <falcon/fassert.h>
#include <falcon/classreference.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/paramerror.h>

namespace Falcon
{

class ClassReference::Reference
{
public:
   Item m_item;
   uint32 mark;

   Reference( const Item& v )
   {
      m_item = v;
      mark = 0;
   }
};   



ClassReference::ClassReference():
   Class("Reference", FLC_ITEM_REF)
{   
}


ClassReference::~ClassReference()
{  
}


void* ClassReference::makeRef( Item& item ) const
{
   static Collector* coll = Engine::instance()->collector();
   
   Reference* r = new Reference( item );
   item.setUser( FALCON_GC_STORE( coll, this, r ) );
   item.content.mth.ref = &r->m_item;
   item.type( FLC_ITEM_REF );
   return r;
}


Class* ClassReference::getParent( const String& ) const
{
   return 0;
}

void ClassReference::dispose( void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   // TODO: pool references
   delete ref;
}


void* ClassReference::clone( void* source ) const
{
   Reference* ref = static_cast<Reference*>(source);
   // TODO: pool references
   return new Reference(*ref);   
}


void ClassReference::serialize( DataWriter* , void*  ) const
{
   // TODO
}


void* ClassReference::deserialize( DataReader* ) const
{
   //TODO
   return 0;
}

   
void ClassReference::gcMark( void* self, uint32 mark ) const
{
   Reference* ref = static_cast<Reference*>(self);
   ref->mark = mark;
   
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->gcMark( data, mark );
   cls->gcMarkMyself( mark );
}


bool ClassReference::gcCheck( void* self, uint32 mark ) const
{
   Reference* ref = static_cast<Reference*>(self);
   return ref->mark >= mark;
}


void ClassReference::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   Reference* ref = static_cast<Reference*>(self);

   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->enumerateProperties( data, cb );
}


void ClassReference::enumeratePV( void* self, PVEnumerator& cb ) const
{
   Reference* ref = static_cast<Reference*>(self);
   
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->enumeratePV( data, cb );
}


bool ClassReference::hasProperty( void* self, const String& prop ) const
{
   Reference* ref = static_cast<Reference*>(self);
   
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   return cls->hasProperty( data, prop );
}


void ClassReference::describe( void* self, String& target, int depth, int maxlen ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );

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
   
   makeRef(ctx->topData());
}


void ClassReference::op_neg( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_neg( ctx, data );
}


void ClassReference::op_add( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_add( ctx, data );   
}


void ClassReference::op_sub( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_sub( ctx, data );      
}


void ClassReference::op_mul( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_mul( ctx, data );   
}


void ClassReference::op_div( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_div( ctx, data );   
}


void ClassReference::op_mod( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_mod( ctx, data );   
}


void ClassReference::op_pow( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_pow( ctx, data );   
}


void ClassReference::op_aadd( VMContext* ctx, void* self) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_aadd( ctx, data );   
}


void ClassReference::op_asub( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_asub( ctx, data );   
}


void ClassReference::op_amul( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_amul( ctx, data );   
}


void ClassReference::op_adiv( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_adiv( ctx, data );   
}


void ClassReference::op_amod( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_amod( ctx, data );   
}


void ClassReference::op_apow( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_apow( ctx, data );   
}


void ClassReference::op_inc( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_inc( ctx, data );   
}


void ClassReference::op_dec(VMContext* ctx, void* self) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_dec( ctx, data );   
}


void ClassReference::op_incpost(VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_incpost( ctx, data );   
}


void ClassReference::op_decpost(VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_decpost( ctx, data );   
}


void ClassReference::op_getIndex(VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_getIndex( ctx, data );   
}


void ClassReference::op_setIndex(VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_setIndex( ctx, data );   
}


void ClassReference::op_getProperty( VMContext* ctx, void* self, const String& prop) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_getProperty( ctx, data, prop );   
}


void ClassReference::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_setProperty( ctx, data, prop );   
}


void ClassReference::op_compare( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_compare( ctx, data );   
}


void ClassReference::op_isTrue( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_isTrue( ctx, data );   
}


void ClassReference::op_in( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_in( ctx, data );   
}


void ClassReference::op_provides( VMContext* ctx, void* self, const String& property ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_provides( ctx, data, property );   
}


void ClassReference::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_call( ctx, paramCount, data );   
}


void ClassReference::op_toString( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_toString( ctx, data );   
}


void ClassReference::op_first( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_first( ctx, data );   
}


void ClassReference::op_next( VMContext* ctx, void* self ) const
{
   Reference* ref = static_cast<Reference*>(self);
   Class* cls;
   void* data;
   ref->m_item.forceClassInst( cls, data );
   cls->op_next( ctx, data );   
}


}

/* end of classreference.cpp */
