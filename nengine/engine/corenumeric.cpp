/*
 FALCON - The Falcon Programming Language.
 FILE: corenumeric.cpp
 
 Function object handler.
 -------------------------------------------------------------------
 Author: Francesco Magliocca
 Begin: Sat, 11 Jun 2011 22:00:05 +0200
 
 -------------------------------------------------------------------
 (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)
 
 See LICENSE file for licensing details.
 */

#include <falcon/corenumeric.h>
#include <falcon/itemid.h>
#include <falcon/item.h>

namespace Falcon {
    

CoreNumeric::CoreNumeric() : Class( "Numeric", FLC_ITEM_NUM ) { }

CoreNumeric::~CoreNumeric() { }

void *CoreNumeric::create( void *creationParams ) const {

   Item *result = new Item;
    
   *result = *static_cast<numeric*>( creationParams );
    
   return result;
    
}


void CoreNumeric::dispose( void *self ) const {
    
   Item *data = (Item*)self;
    
   delete data;
    
}

void *CoreNumeric::clone( void *source ) const {
    
   Item *result = new Item;
    
   *result = *static_cast<Item*>( source );
    
   return result;
    
}

void CoreNumeric::serialize( DataWriter*, void* ) const {
    
   //TODO
    
}


void* CoreNumeric::deserialize( DataReader* ) const {
    
   //TODO
   return 0;
    
}

void CoreNumeric::describe( void* instance, String& target, int, int  ) const {
    
   target.N(((Item*) instance)->asNumeric() );
    
}

// ================================================================

void CoreNumeric::op_add( VMachine *, void* ) const {
    
   // TODO
    
}

void CoreNumeric::op_sub( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_mul( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_div( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_mod( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_pow( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_aadd( VMachine *, void*) const {
    
   // TODO
    
}

void CoreNumeric::op_asub( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_amul( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_adiv( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_amod( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_apow( VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_inc(VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_dec(VMachine *, void*) const {
    
   // TODO
    
}


void CoreNumeric::op_incpost(VMachine *, void* ) const {
    
   // TODO
    
}


void CoreNumeric::op_decpost(VMachine *, void* ) const {
    
   // TODO
    
}


}
