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

   Item m_itemNil;
   bool m_bHasNil;
   
   Item m_itemTrue;
   bool m_bHasTrue;
   
   Item m_itemFalse;
   bool m_bHasFalse;
   
   RangeMap m_rangeMap;
   IntegerMap m_intMap;
   StringMap m_stringMap;
   InstanceMap m_instMap;
   
   Private() {}
   
   Private( const Private& other):
      m_bHasNil( false ),
      m_bHasTrue( false ),
      m_bHasFalse( false ),
      
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
      case FLC_ITEM_NIL:
         _p->m_bHasNil = true;
         _p->m_itemNil = value;
         break;
         
      case FLC_ITEM_BOOL:
         if( key.asBoolean() )
         {
            _p->m_bHasTrue = true;
            _p->m_itemTrue = value;
         }
         else
         {
            _p->m_bHasFalse = true;
            _p->m_itemFalse = value;
         }
         break;
         
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
      case FLC_ITEM_NIL:
         _p->m_bHasNil = false;
         break;
         
      case FLC_ITEM_BOOL:
         if( key.asBoolean() )
         {
            _p->m_bHasTrue = false;
         }
         else
         {
            _p->m_bHasFalse = false;
         }
         break;
         
      case FLC_ITEM_INT:
         _p->m_intMap.erase(key.asInteger());
         break;
         
      case FLC_ITEM_NUM:
         _p->m_intMap.erase( (int64) key.asNumeric() );
         break;
         
      default:
      {
         Class *cls;
         void* data;
         
         key.forceClassInst( cls, data );
         switch( cls->typeID() )
         {
            case FLC_CLASS_ID_STRING:
               _p->m_stringMap.erase( *static_cast<String*>(data) );
               break;
               
            case FLC_CLASS_ID_RANGE:
               _p->m_rangeMap.erase( *static_cast<Range*>(data) );
               break;
               
            default:
               _p->m_instMap.erase( 
                  Private::class_data_pair( cls, data ) );
               break;
         }
      }
      
   }
}


Item* ItemDict::find( const Item& key )
{

   switch( key.dereference()->type() )
   {      
      case FLC_ITEM_NIL:
         if( _p->m_bHasNil )
         {
            return &_p->m_itemNil;
         }
         return 0;
         
      case FLC_ITEM_BOOL:
         if( key.asBoolean() )
         {
            if( _p->m_bHasTrue )
            {
               return &_p->m_itemTrue;
            }
         }
         else
         {
            if( _p->m_bHasFalse )
            {
               return &_p->m_itemFalse;
            }
         }
         return 0;
         
      case FLC_ITEM_INT:
      {
         Private::IntegerMap::iterator iter = _p->m_intMap.find(key.asInteger());
         if( iter != _p->m_intMap.end() )
         {
            return &iter->second;
         }
      }
      return 0;
         
      case FLC_ITEM_NUM:
      {
         Private::IntegerMap::iterator iter = _p->m_intMap.find((int64) key.asNumeric());
         if( iter != _p->m_intMap.end() )
         {
            return &iter->second;
         }
      }
      return 0;               
      
      default:
      {
         Class *cls;
         void* data;
         
         key.forceClassInst( cls, data );
         switch( cls->typeID() )
         {
            case FLC_CLASS_ID_STRING:
            {
               Private::StringMap::iterator iter = _p->m_stringMap.find(*key.asString());
               if( iter != _p->m_stringMap.end() )
               {
                  return &iter->second;
               }
            }
            return 0;                       
               
            case FLC_CLASS_ID_RANGE:
            {
               Private::RangeMap::iterator iter = _p->m_rangeMap.find(*(Range*)key.asInst());
               if( iter != _p->m_rangeMap.end() )
               {
                  return &iter->second;
               }
            }
            return 0;                       
               
            default:
            {
               Private::InstanceMap::iterator iter = _p->m_instMap.find(
                  Private::class_data_pair( cls, data ));
               if( iter != _p->m_instMap.end() )
               {
                  return &iter->second;
               }
            }
            break;
         }
      }
   }
   
   return 0;                       
}


length_t ItemDict::size() const
{
   length_t count = 0;
   if( _p->m_bHasNil ) ++count;
   if( _p->m_bHasTrue ) ++count;
   if( _p->m_bHasFalse ) ++count;
   
   count += _p->m_intMap.size();
   count += _p->m_rangeMap.size();
   count += _p->m_stringMap.size();
   count += _p->m_instMap.size();
   
   return count;
}


void ItemDict::describe( String& target, int depth, int maxlen ) const
{
   String ks, vs;
   
   if( depth == 0 )
   {
      target = "...";
      return;
   }
   
   target.size(0);
   target += "[";

   if( _p->m_bHasNil )
   {
      if( target.size() > 1 )
      {
         target += ", ";
      }
      
      vs.size(0);
      _p->m_itemNil.describe( vs, depth-1, maxlen );
      target += "nil => ";
      target += vs;
   }
   

   if( _p->m_bHasTrue )
   {
      if( target.size() > 1 )
      {
         target += ", ";
      }
      
      vs.size(0);
      _p->m_itemTrue.describe( vs, depth-1, maxlen );
      target += "true => ";
      target += vs;
   }

   if( _p->m_bHasFalse )
   {
      if( target.size() > 1 )
      {
         target += ", ";
      }
      
      vs.size(0);
      _p->m_itemFalse.describe( vs, depth-1, maxlen );
      target += "false => ";
      target += vs;
   }
   
   Private::IntegerMap::iterator iiter = _p->m_intMap.begin();   
   while( iiter != _p->m_intMap.end() )
   {
      const Item& value = iiter->second;
      value.describe( vs, depth-1, maxlen );
      if( target.size() > 1 )
      {
         target += ", ";
      }

      target.N( iiter->first ).A( " => ").A( vs );
      ++iiter;
   }

   Private::RangeMap::iterator riter = _p->m_rangeMap.begin();   
   while( riter != _p->m_rangeMap.end() )
   {
      const Item& value = riter->second;      
      value.describe( vs, depth-1, maxlen );
      if( target.size() > 1 )
      {
         target += ", ";
      }

      target += riter->first.describe() + " => " + vs;
      ++riter;
   }
   
   Private::StringMap::iterator siter = _p->m_stringMap.begin();   
   while( siter != _p->m_stringMap.end() )
   {
      const Item& value = iiter->second;      
      value.describe( vs, depth-1, maxlen );
      if( target.size() > 1 )
      {
         target += ", ";
      }

      target += siter->first + " => " + vs;
      ++siter;
   }
   
   Private::InstanceMap::iterator kiter = _p->m_instMap.begin();   
   while( kiter != _p->m_instMap.end() )
   {
      const Private::class_data_pair& pair = kiter->first;
      const Item& value = kiter->second;
      
      pair.cls->describe( pair.data, ks, depth-1, maxlen );
      value.describe( vs, depth-1, maxlen );
      if( target.size() > 1 )
      {
         target += ", ";
      }

      target += ks + " => " + vs;
      ++kiter;
   }
   
   target += "]";
}


void ItemDict::enumerate( Enumerator& rator )
{
   static Class* cstr = Engine::instance()->stringClass();
   static Class* cr = Engine::instance()->rangeClass();
   
   if( _p->m_bHasNil ) rator( Item(), _p->m_itemNil );
   if( _p->m_bHasTrue ) rator( Item( true ), _p->m_itemTrue );
   if( _p->m_bHasFalse ) rator( Item( false ), _p->m_itemFalse );
  
   Private::IntegerMap::iterator iiter = _p->m_intMap.begin();   
   while( iiter != _p->m_intMap.end() )
   {
      rator( iiter->first, iiter->second );      
      ++iiter;
   }

   Private::RangeMap::iterator riter = _p->m_rangeMap.begin();   
   while( riter != _p->m_rangeMap.end() )
   {
      void* data = const_cast<Range*>( &riter->first );
      rator( Item( cr, data ), riter->second );      
      ++riter;
   }
   
   Private::StringMap::iterator siter = _p->m_stringMap.begin();   
   while( siter != _p->m_stringMap.end() )
   {
      void* data = const_cast<String*>( &siter->first );
      rator( Item( cstr, data ) , siter->second );      
      ++siter;
   }
   
   Private::InstanceMap::iterator kiter = _p->m_instMap.begin();   
   while( kiter != _p->m_instMap.end() )
   {
      const Private::class_data_pair& pair = kiter->first;
      rator( Item(pair.cls, pair.data), kiter->second );      
      ++kiter;
   }
}

}

/* end of itemdict.cpp */
