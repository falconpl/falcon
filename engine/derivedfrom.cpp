/*
   FALCON - The Falcon Programming Language.
   FILE: derivedfrom.h

   Class implementing common behavior for classes with a single parent.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 20:54:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/derivedfrom.cpp"

#include <falcon/derivedfrom.h>

namespace Falcon {


DerivedFrom::DerivedFrom( Class* parent, const String& name ):
   Class(name),
   m_parent( parent )
{}


DerivedFrom::~DerivedFrom() 
{}


bool DerivedFrom::isDerivedFrom( Class* cls ) const
{
   return cls == this || m_parent->isDerivedFrom(cls);
}

Class* DerivedFrom::getParent( const String& name ) const
{
   if( name == m_parent->name() ) return m_parent;
   return 0;
}


void* DerivedFrom::getParentData( Class* parent, void* data ) const
{
   if( parent == m_parent || parent == this ) return data;
   return 0;
}


void DerivedFrom::enumerateParents( Class::ClassEnumerator& cb ) const
{
   cb( m_parent, true );
}


void DerivedFrom::enumerateProperties( void* instance, Class::PropertyEnumerator& cb ) const
{
   m_parent->enumerateProperties( instance, cb );
}

void DerivedFrom::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{
   m_parent->enumeratePV( instance, cb );
}

bool DerivedFrom::hasProperty( void* instance, const String& prop ) const
{
   return m_parent->hasProperty( instance, prop );
}

bool DerivedFrom::op_getParentProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   // screens parent's parent
   if( m_parent->hasProperty( prop ) )
   {
      m_parent->op_getProperty( ctx, instance, prop );
      return true;
   }
   return false;
}


void DerivedFrom::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   if( prop == m_parent->name() )
   {
      ctx->stackResult( 2, Item( m_parent, instance ) );
   }
   else
   {
      if( m_parent->hasProperty( prop ) )
      {
         m_parent->op_getProperty( ctx, instance, prop );
      }
      else
      {
         Class::op_getProperty( ctx, instance, prop );
      }
   }
   
}
   
 
void DerivedFrom::op_setProperty( VMContext* ctx, void* instance, const String& prop) const
{
   m_parent->op_setProperty( ctx, instance, prop );
}


void DerivedFrom::dispose( void* instance ) const
{
   m_parent->dispose( instance );
}


void* DerivedFrom::clone( void* instance ) const
{
   return m_parent->clone( instance );
}

void DerivedFrom::store( VMContext* ctx, DataWriter* stream, void* instance ) const
{
   m_parent->store( ctx, stream, instance );
}

void DerivedFrom::restore( VMContext* ctx, DataReader* stream, void*& empty ) const
{
   m_parent->restore( ctx, stream, empty );
}

void DerivedFrom::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   m_parent->flatten( ctx, subItems, instance );
}

void DerivedFrom::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   m_parent->flatten( ctx, subItems, instance );
}

//=========================================================
// Class management
//

void DerivedFrom::gcMark( void* instance, uint32 mark ) const
{
   m_parent->gcMark( instance, mark );
}

bool DerivedFrom::gcCheck( void* instance, uint32 mark ) const
{
   return m_parent->gcCheck( instance, mark );
}

//=========================================================
// Operators.
//

void DerivedFrom::op_neg( VMContext* ctx, void* instance ) const
{
   return m_parent->op_neg( ctx, instance );
}

void DerivedFrom::op_add( VMContext* ctx, void* instance ) const
{
   return m_parent->op_add( ctx, instance );
}

void DerivedFrom::op_sub( VMContext* ctx, void* instance ) const
{
   return m_parent->op_sub( ctx, instance );
}


void DerivedFrom::op_mul( VMContext* ctx, void* instance ) const
{
   return m_parent->op_mul( ctx, instance );
}


void DerivedFrom::op_div( VMContext* ctx, void* instance ) const
{
   return m_parent->op_div( ctx, instance );
}


void DerivedFrom::op_mod( VMContext* ctx, void* instance ) const
{
   return m_parent->op_mod( ctx, instance );
}


void DerivedFrom::op_pow( VMContext* ctx, void* instance ) const
{
   return m_parent->op_pow( ctx, instance );
}


void DerivedFrom::op_shr( VMContext* ctx, void* instance ) const
{
   return m_parent->op_shr( ctx, instance );
}


void DerivedFrom::op_shl( VMContext* ctx, void* instance ) const
{
   return m_parent->op_shl( ctx, instance );
}


void DerivedFrom::op_aadd( VMContext* ctx, void* instance) const
{
   return m_parent->op_aadd( ctx, instance );
}


void DerivedFrom::op_asub( VMContext* ctx, void* instance ) const
{
   return m_parent->op_asub( ctx, instance );
}


void DerivedFrom::op_amul( VMContext* ctx, void* instance ) const
{
   return m_parent->op_amul( ctx, instance );
}


void DerivedFrom::op_adiv( VMContext* ctx, void* instance ) const
{
   return m_parent->op_adiv( ctx, instance );
}


void DerivedFrom::op_amod( VMContext* ctx, void* instance ) const
{
   return m_parent->op_amod( ctx, instance );
}


void DerivedFrom::op_apow( VMContext* ctx, void* instance ) const
{
   return m_parent->op_apow( ctx, instance );
}


void DerivedFrom::op_ashr( VMContext* ctx, void* instance ) const
{
   return m_parent->op_ashr( ctx, instance );
}

void DerivedFrom::op_ashl( VMContext* ctx, void* instance ) const
{
   return m_parent->op_ashl( ctx, instance );
}


void DerivedFrom::op_inc( VMContext* ctx, void* instance ) const
{
   return m_parent->op_inc( ctx, instance );
}


void DerivedFrom::op_dec(VMContext* ctx, void* instance) const
{
   return m_parent->op_dec( ctx, instance );
}


void DerivedFrom::op_incpost(VMContext* ctx, void* instance ) const
{
   return m_parent->op_incpost( ctx, instance );
}


void DerivedFrom::op_decpost(VMContext* ctx, void* instance ) const
{
   return m_parent->op_decpost( ctx, instance );
}

void DerivedFrom::op_getIndex(VMContext* ctx, void* instance ) const
{
   return m_parent->op_getIndex( ctx, instance );
}


void DerivedFrom::op_setIndex(VMContext* ctx, void* instance ) const
{
   return m_parent->op_setIndex( ctx, instance );
}


void DerivedFrom::op_compare( VMContext* ctx, void* instance ) const
{
   return m_parent->op_compare( ctx, instance );
}


void DerivedFrom::op_isTrue( VMContext* ctx, void* instance ) const
{
   return m_parent->op_isTrue( ctx, instance );
}


void DerivedFrom::op_in( VMContext* ctx, void* instance ) const
{
   return m_parent->op_in( ctx, instance );
}


void DerivedFrom::op_provides( VMContext* ctx, void* instance, const String& property ) const
{
   return m_parent->op_provides( ctx, instance );
}


void DerivedFrom::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   return m_parent->op_call( ctx, instance );
}

void DerivedFrom::op_toString( VMContext* ctx, void* instance ) const
{
   return m_parent->op_toString( ctx, instance );
}

void DerivedFrom::op_iter( VMContext* ctx, void* instance ) const
{
   return m_parent->op_iter( ctx, instance );
}

void DerivedFrom::op_next( VMContext* ctx, void* instance ) const
{
   return m_parent->op_next( ctx, instance );
}


}

/* end of derivedfrom.cpp */
