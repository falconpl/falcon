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


bool CoreDict::hasProperty( void* self, const String& prop ) const
{
   if( prop == "len" ) return true;
   return false;
}

bool CoreDict::getProperty( void* self, const String& property, Item& value ) const
{
   if( property == "len" )
   {
      value.setInteger(static_cast<ItemDictionary*>(self)->size());
      return true;
   }

   return false;
}

bool CoreDict::getIndex( void* self, const Item& index, Item& value ) const
{
   ItemDictionary& dict = *static_cast<ItemDictionary*>(self);
   ItemDictionary::iterator pos = dict.find(index);
   if( pos != dict.end() )
   {
      value = pos->second;
      return true;
   }
   
   return false;
}

bool CoreDict::setIndex( void* self, const Item& index, const Item& value ) const
{
   ItemDictionary& dict = *static_cast<ItemDictionary*>(self);
   dict[index] = value;
   return true;
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

void CoreDict::op_add( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   //TODO
}

void CoreDict::op_isTrue( VMachine *vm, void* self, Item& target ) const
{
   target.setBoolean( static_cast<ItemDictionary*>(self)->size() != 0 );
}

void CoreDict::op_toString( VMachine *vm, void* self, Item& target ) const
{
   // todo
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
