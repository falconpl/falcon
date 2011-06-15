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


void CoreDict::serialize( DataWriter*, void* ) const
{
   // todo
}


void* CoreDict::deserialize( DataReader* ) const
{
   //todo
   return 0;
}

void CoreDict::describe( void* instance, String& target, int maxDepth, int maxLen ) const
{
   if( maxDepth == 0 )
   {
      target = "...";
      return;
   }
   
   ItemDictionary* dict = static_cast<ItemDictionary*>(instance);
   target.size(0);
   ItemDictionary::iterator iter = dict->begin();
   String temp;
   while( iter != dict->end() )
   {
      if( target.size() == 0 )
      {
         target += "[";
      }
      else
      {
         target += ", ";
      }

      Class* cls;
      void* inst;

      temp.size(0);
      iter->first.forceClassInst(cls, inst);
      cls->describe( inst, temp, maxDepth - 1, maxLen );
      target += temp;
      target += " => ";

      temp.size(0);
      iter->second.forceClassInst(cls, inst);
      cls->describe( inst, temp, maxDepth - 1, maxLen );
      target += temp;
   }

   target += "]";
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

void CoreDict::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   cb("len", true);
}

//=======================================================================
//

void CoreDict::op_add( VMachine *, void* ) const
{
   //TODO
}

void CoreDict::op_isTrue( VMachine *vm, void* self ) const
{
   vm->stackResult( 1, static_cast<ItemDictionary*>(self)->size() != 0 );
}

void CoreDict::op_toString( VMachine *vm, void* self ) const
{
   String s;
   s.A("[Dictionary of ").N((int64)static_cast<ItemDictionary*>(self)->size()).A(" elements]");
   vm->stackResult( 1, s );
}


void CoreDict::op_getProperty( VMachine *vm, void* self, const String& property ) const
{
   Class::op_getProperty( vm, self, property );
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


}

/* end of coredict.cpp */
