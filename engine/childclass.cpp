/*
   FALCON - The Falcon Programming Language.
   FILE: childclass.cpp

   Simple implementation of a child class for native classes.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Jul 2014 18:19:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/childclass.h"

#include <falcon/childclass.h>

namespace Falcon {

ChildClass::ChildClass( const String& name, const Class* parent, int64 typeID ):
            Class( name, typeID < 0 ? parent->typeID() : typeID)
{
   setParent(parent);

   m_bIsFalconClass = parent->isFalconClass();
   m_bIsErrorClass = parent->isErrorClass();
   m_bIsFlatInstance = parent->isFlatInstance();
   m_bHasSharedInstances = parent->hasSharedInstances();
   m_userFlags = parent->userFlags();
   m_typeID = parent->typeID();
   m_clearPriority = parent->clearPriority();
}

ChildClass::~ChildClass()
{}
   
void ChildClass::render( TextWriter* tw, int32 depth ) const
{
   m_parent->render(tw, depth);
}

int64 ChildClass::occupiedMemory( void* instance ) const
{
   return m_parent->occupiedMemory(instance);
}


void ChildClass::dispose( void* instance ) const
{
   m_parent->dispose(instance);
}

void* ChildClass::clone( void* instance ) const
{
   return m_parent->clone(instance);
}


void* ChildClass::createInstance() const
{
   return m_parent->createInstance();
}


void ChildClass::store( VMContext* ctx, DataWriter* stream, void* instance ) const
{
   m_parent->store(ctx, stream, instance);
}

void ChildClass::restore( VMContext* ctx, DataReader* stream ) const
{
   m_parent->restore( ctx, stream );
}

void ChildClass::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   m_parent->flatten(ctx, subItems, instance );
}

void ChildClass::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   m_parent->unflatten(ctx, subItems, instance);
}


void ChildClass::delegate( void* instance, Item* target, const String& message ) const
{
   m_parent->delegate( instance, target, message );
}
        
void ChildClass::gcMarkInstance( void* instance, uint32 mark ) const
{
   m_parent->gcMarkInstance(instance, mark);
}

bool ChildClass::gcCheckInstance( void* instance, uint32 mark ) const
{
   return m_parent->gcCheckInstance(instance, mark);
}

void ChildClass::describe( void* instance, String& target, int depth, int maxlen ) const
{
   m_parent->describe( instance, target, depth, maxlen );
}


void ChildClass::inspect( void* instance, String& target, int depth ) const
{
   m_parent->inspect( instance, target, depth );
}
        
bool ChildClass::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
  return m_parent->op_init(ctx, instance, pcount);
}

void ChildClass::op_neg( VMContext* ctx, void* instance ) const
{
   m_parent->op_neg( ctx, instance );
}

void ChildClass::op_add( VMContext* ctx, void* instance ) const
{
   m_parent->op_add( ctx, instance );
}

void ChildClass::op_sub( VMContext* ctx, void* instance ) const
{
   m_parent->op_sub( ctx, instance );
}

void ChildClass::op_mul( VMContext* ctx, void* instance ) const
{
   m_parent->op_mul( ctx, instance );
}

void ChildClass::op_div( VMContext* ctx, void* instance ) const
{
   m_parent->op_div( ctx, instance );
}

void ChildClass::op_mod( VMContext* ctx, void* instance ) const
{
   m_parent->op_mod( ctx, instance );
}

void ChildClass::op_pow( VMContext* ctx, void* instance ) const
{
   m_parent->op_pow( ctx, instance );
}

void ChildClass::op_shr( VMContext* ctx, void* instance ) const
{
   m_parent->op_shr( ctx, instance );
}

void ChildClass::op_shl( VMContext* ctx, void* instance ) const
{
   m_parent->op_shl( ctx, instance );
}

void ChildClass::op_aadd( VMContext* ctx, void* instance) const
{
   m_parent->op_aadd( ctx, instance );
}

void ChildClass::op_asub( VMContext* ctx, void* instance ) const
{
   m_parent->op_asub( ctx, instance );
}

void ChildClass::op_amul( VMContext* ctx, void* instance ) const
{
   m_parent->op_amul( ctx, instance );
}

void ChildClass::op_adiv( VMContext* ctx, void* instance ) const
{
   m_parent->op_adiv( ctx, instance );
}

void ChildClass::op_amod( VMContext* ctx, void* instance ) const
{
   m_parent->op_amod( ctx, instance );
}

void ChildClass::op_apow( VMContext* ctx, void* instance ) const
{
   m_parent->op_apow( ctx, instance );
}

void ChildClass::op_ashr( VMContext* ctx, void* instance ) const
{
   m_parent->op_ashr( ctx, instance );
}

void ChildClass::op_ashl( VMContext* ctx, void* instance ) const
{
   m_parent->op_ashl( ctx, instance );
}

void ChildClass::op_inc( VMContext* ctx, void* instance ) const
{
   m_parent->op_inc( ctx, instance );
}

void ChildClass::op_dec(VMContext* ctx, void* instance) const
{
   m_parent->op_dec( ctx, instance );
}

void ChildClass::op_incpost(VMContext* ctx, void* instance ) const
{
   m_parent->op_incpost( ctx, instance );
}

void ChildClass::op_decpost(VMContext* ctx, void* instance ) const
{
   m_parent->op_decpost( ctx, instance );
}

void ChildClass::op_getIndex(VMContext* ctx, void* instance ) const
{
   m_parent->op_getIndex( ctx, instance );
}

void ChildClass::op_setIndex(VMContext* ctx, void* instance ) const
{
   m_parent->op_setIndex( ctx, instance );
}

void ChildClass::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   m_parent->op_getProperty( ctx, instance, prop );
}

void ChildClass::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   m_parent->op_setProperty( ctx, instance, prop );
}

void ChildClass::op_getClassProperty( VMContext* ctx, const String& prop) const
{
   m_parent->op_getClassProperty( ctx, prop );
}

void ChildClass::op_setClassProperty( VMContext* ctx, const String& prop ) const
{
   m_parent->op_setClassProperty( ctx, prop );
}

void ChildClass::op_compare( VMContext* ctx, void* instance ) const
{
   m_parent->op_compare( ctx, instance );
}

void ChildClass::op_isTrue( VMContext* ctx, void* instance ) const
{
   m_parent->op_isTrue( ctx, instance );
}

void ChildClass::op_in( VMContext* ctx, void* instance ) const
{
   m_parent->op_in( ctx, instance );
}

void ChildClass::op_provides( VMContext* ctx, void* instance, const String& propName ) const
{
   m_parent->op_provides( ctx, instance, propName );
}

void ChildClass::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   m_parent->op_call( ctx, paramCount, instance );
}

void ChildClass::op_toString( VMContext* ctx, void* instance ) const
{
   m_parent->op_toString( ctx, instance );
}

void ChildClass::op_iter( VMContext* ctx, void* instance ) const
{
   m_parent->op_iter( ctx, instance );
}

void ChildClass::op_next( VMContext* ctx, void* instance ) const
{
   m_parent->op_next( ctx, instance );
}

void ChildClass::op_summon( VMContext* ctx, void* instance, const String& message, int32 pCount, bool bOptional ) const
{
   m_parent->op_summon( ctx, instance, message, pCount, bOptional );
}

void ChildClass::op_summon_failing( VMContext* ctx, void* instance, const String& message, int32 pCount ) const
{
   m_parent->op_summon_failing( ctx, instance, message, pCount );
}



Selectable* ChildClass::getSelectableInterface( void* instance ) const
{
   return m_parent->getSelectableInterface( instance );
}

}

/* childclass.cpp */

