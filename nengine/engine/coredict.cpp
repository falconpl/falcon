/*
   FALCON - The Falcon Programming Language.
   FILE: coredict.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 15:33:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/coredict.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>

#include <map>

#include "falcon/accesserror.h"

namespace Falcon {

typedef std::map<Item, Item> ItemDictionary;

CoreDict::CoreDict():
   Class("Dictionary", FLC_CLASS_ID_DICT )
{
}


CoreDict::~CoreDict()
{
}


void* CoreDict::create(void* creationParams ) const
{
   // Shall we copy a creator?
   cpars* cp = static_cast<cpars*>(creationParams);
   if( cp->m_other != 0 )
   {
      if ( cp->m_bCopy )
         return new ItemDictionary( *static_cast<ItemDictionary*>(cp->m_other) );
      else
         return cp->m_other;
   }

   // -- or shall we just generate a new array?
   return new ItemDictionary;
}


void CoreDict::dispose( void* self ) const
{
   ItemDictionary* f = static_cast<ItemDictionary*>(self);
   delete f;
}


void* CoreDict::clone( void* source ) const
{
   ItemDictionary* array = static_cast<ItemDictionary*>(source);
   return new ItemDictionary(*array);
}


void CoreDict::serialize( DataWriter* stream, void* self ) const
{
   // todo
}


void* CoreDict::deserialize( DataReader* stream ) const
{
   //todo
   return 0;
}

void CoreDict::describe( void* instance, String& target ) const
{
   ItemDictionary* arr = static_cast<ItemDictionary*>(instance);
   target = String("[Dict of ").N((int64)arr->size()).A(" elements]");
}



void CoreDict::gcMark( void* self, uint32 mark ) const
{
   ItemDictionary& dict = *static_cast<ItemDictionary*>(self);
   ItemDictionary::iterator pos = dict.begin();
   while( pos != dict.end() )
   {
      {
         const Item& tgt = pos->first;
         if( tgt.isDeep() )
         {
            tgt.asDeepClass()->gcMark(tgt.asDeepInst(), mark);
         }
      }

      {
         Item& tgt = pos->second;
         if( tgt.isDeep() )
         {
            tgt.asDeepClass()->gcMark(tgt.asDeepInst(), mark);
         }
      }

      ++pos;
   }
}

void CoreDict::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   cb("len", true);
}

//=======================================================================
//

void CoreDict::op_add( VMachine *vm, void* self ) const
{
   //TODO
}

void CoreDict::op_isTrue( VMachine *vm, void* self ) const
{
   vm->stackResult( 1, static_cast<ItemDictionary*>(self)->size() != 0 );
}

void CoreDict::op_toString( VMachine *vm, void* self ) const
{
   // todo
}


void CoreDict::op_getProperty( VMachine *vm, void* self, const String& property ) const
{
   if( property == "len" )
   {
      vm->stackResult( 1, (int64) static_cast<ItemDictionary*>(self)->size() );
   }
   else
   {
      throw new AccessError( ErrorParam( e_prop_acc, __LINE__, __FILE__ ).extra(property) );
   }
}

void CoreDict::op_getIndex( VMachine* vm, void* self ) const
{
   Item *index, *dict_item;
   vm->operands( index, dict_item );

   ItemDictionary& dict = *static_cast<ItemDictionary*>(self);
   ItemDictionary::iterator pos = dict.find(*index);
   
   if( pos != dict.end() )
   {
      vm->stackResult( 2, pos->second );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__, __FILE__ ) );
   }
}

void CoreDict::op_setIndex( VMachine* vm, void* self ) const
{
   Item *value, *index, *dict_item;
   vm->operands( value, index, dict_item );

   ItemDictionary& dict = *static_cast<ItemDictionary*>(self);
   dict[*index] = *value;
   vm->stackResult(3, *value);
}


CoreDict::ToStringNextOp::ToStringNextOp()
{
   apply = apply_;
}

void CoreDict::ToStringNextOp::apply_( const PStep*step, VMachine* vm )
{
   //todo
}

}

/* end of coredict.cpp */
