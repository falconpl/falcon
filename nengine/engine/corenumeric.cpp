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

void CoreNumeric::serialize( DataWriter* stream, void* self ) const {
    
   //TODO
    
}


void* CoreNumeric::deserialize( DataReader* stream ) const {
    
   //TODO
   return 0;
    
}

void CoreNumeric::describe( void* instance, String& target, int, int  ) const {
    
   target.N(((Item*) instance)->asNumeric() );
    
}

// ================================================================

void CoreNumeric::op_add( VMachine *vm, void* self ) const {
    
   // TODO
    
}

void CoreNumeric::op_sub( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_mul( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_div( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_mod( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_pow( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_aadd( VMachine *vm, void* self) const {
    
   // TODO
    
}

void CoreNumeric::op_asub( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_amul( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_adiv( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_amod( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_apow( VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_inc(VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_dec(VMachine *vm, void* self) const {
    
   // TODO
    
}


void CoreNumeric::op_incpost(VMachine *vm, void* self ) const {
    
   // TODO
    
}


void CoreNumeric::op_decpost(VMachine *vm, void* self ) const {
    
   // TODO
    
}


}
