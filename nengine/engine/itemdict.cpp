/*
   FALCON - The Falcon Programming Language.
   FILE: itemdict.cpp

   Class storing lexicographic ordered item dictionaries.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 18 Jul 2011 02:22:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/itemdict.h>
#include <falcon/item.h>
#include <falcon/range.h>
#include <falcon/class.h>
#include <falcon/itemid.h>
#include <falcon/accesstypeerror.h>

#include <map>

namespace Falcon
{

class ItemDict::Private
{
public:   
   typedef std::map<int64, Item> IntegerMap;
   typedef std::map<String, Item> StringMap;

   class class_data_pair 
   {
   public:
      Class* cls;
      void* data;
      
      class_data_pair():
         cls(0),
         data(0)
      {}
      
      class_data_pair( Class* c, void* d ):
         cls(c),
         data(d)
      {}
   };
   
   class cdcomparer
   {
   public:
      bool operator()( const class_data_pair& first, const class_data_pair& second )
      {
         if( first.cls < second.cls ) return true;
         if( first.cls > second.cls ) return false;
         return first.data < second.data;
      }
   };
   
   class rangecomparer
   {
   public:
      bool operator()( const Range& first, const Range& second )
      {
         return first.compare(second) < 0;
      }
   };
   
   typedef std::map<class_data_pair, Item, cdcomparer> InstanceMap;
   typedef std::map<Range, Item, rangecomparer> RangeMap;

   
   RangeMap m_rangeMap;
   IntegerMap m_intMap;
   StringMap m_stringMap;
   InstanceMap m_instMap;
   
   Private() {}
   
   Private( const Private& other):
      m_rangeMap( other.m_rangeMap ),
      m_intMap( other.m_intMap ),
      m_stringMap( other.m_stringMap ),
      m_instMap( other.m_instMap )
   {}
   
   ~Private() {}
   
   void gcMark( uint32 mark )
   {

      RangeMap::iterator irange = m_rangeMap.begin();
      while( irange != m_rangeMap.end() )
      {
         irange->second.gcMark( mark );
         ++irange;
      }
      
      IntegerMap::iterator iint = m_intMap.begin();
      while( iint != m_intMap.end() )
      {
         iint->second.gcMark( mark );
         ++iint;
      }
      
      StringMap::iterator istring = m_stringMap.begin();
      while( istring != m_stringMap.end() )
      {
         istring->second.gcMark( mark );
         ++istring;
      }
      
      InstanceMap::iterator iinst = m_instMap.begin();
      while( iinst != m_instMap.end() )
      {
         iinst->first.cls->gcMark( iinst->first.data, mark );
         iinst->first.cls->gcMarkMyself( mark );
         iinst->second.gcMark( mark );
         ++iinst;
      }
   }

};


ItemDict::ItemDict():
   _p( new Private ),
   m_flags(0),
   m_currentMark(0)
{}


ItemDict::ItemDict( const ItemDict& other ):
   _p( new Private(*other._p) ),
   m_flags( other.m_flags ),
   m_currentMark( other.m_currentMark )
{}


ItemDict::~ItemDict()
{
   delete _p;
}

void ItemDict::gcMark( uint32 mark )
{
   if( m_currentMark >= mark )
   {
      return;
   }

   m_currentMark = mark;
   _p->gcMark( mark );
}


void ItemDict::insert( const Item& key, const Item& value )
{  
   if( key.isReference() )
   {
      insert( *key.dereference(), value );
      return;
   }
   
   switch( key.type() )
   {      
      case FLC_ITEM_INT:
         _p->m_intMap[ key.asInteger() ] = value;
         break;
         
      case FLC_ITEM_NUM:
         _p->m_intMap[ (int64) key.asNumeric() ] = value;
         break;
                       
      default:
      {
         Class *cls;
         void* data;
         
         key.forceClassInst( cls, data );
         switch( cls->typeID() )
         {
            case FLC_CLASS_ID_STRING:
               _p->m_stringMap[ *static_cast<String*>(data) ] = value;
               break;
               
            case FLC_CLASS_ID_RANGE:
               _p->m_rangeMap[ *static_cast<Range*>(data) ] = value;
               break;
               
            default:
               _p->m_instMap[ 
                  Private::class_data_pair( cls, data ) ] = value;
               break;
         }
      }
   }
}


void ItemDict::remove( const Item& key )
{
   if( key.isReference() )
   {
      remove( *key.dereference() );
      return;
   }
   
   switch( key.type() )
   {      
      case FLC_ITEM_INT:
         _p->m_intMap.erase(key.asInteger());
         break;
         
      case FLC_ITEM_NUM:
         _p->m_intMap.erase( (int64) key.asNumeric() );
         break;
         
         
      case FLC_ITEM_USER:
         _p->m_instMap.erase(
              Private::class_data_pair( key.asClass(), key.asInst() ) );
         break;
   
      default:
         // should not happen, but...
         throw new AccessTypeError( ErrorParam( e_dict_key, __LINE__, SRC ) );
   }
}


Item* ItemDict::find( const Item&  )
{
   return 0;
}


length_t ItemDict::size() const
{
   return 0;
}


void ItemDict::describe( String& target, int depth, int  ) const
{
   if( depth == 0 )
   {
      target = "...";
      return;
   }
   
   /*
   Private::ItemMap& dict = _p->m_items;
   Private::ItemMap::const_iterator pos = dict.begin();

   target.size(0);
   target += "[";
   while( pos != dict.end() )
   {
      const Item& key = pos->first;
      const Item& value = pos->second;

      String ks, vs;
      key.describe( ks, depth-1, maxlen );
      value.describe( vs, depth-1, maxlen );
      if( target.size() > 1 )
      {
         target += ", ";
      }

      target += ks + " => " + vs;
      ++pos;
   }

   target += "]";
    */
}


void ItemDict::enumerate( Enumerator&  )
{
   
}


}

/* end of itemdict.cpp */
