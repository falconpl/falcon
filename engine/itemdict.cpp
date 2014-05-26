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

#undef SRC
#define SRC "engine/itemdict.cpp"

#include <falcon/itemdict.h>
#include <falcon/item.h>
#include <falcon/range.h>
#include <falcon/class.h>
#include <falcon/itemid.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>

#include <falcon/stderrors.h>

#include <map>

namespace Falcon
{

class ItemDict::Private
{
public:
   class ItemComparer
   {
   public:
      bool operator()( const Item& first, const Item& second ) const
      {
         return first.compare(second) < 0;
      }
   };
   
   typedef std::map<Item, Item, ItemComparer> ItemMap;

   ItemMap m_itemMap;
   
   Private()
   {}
   
   Private( const Private& other):
      m_itemMap( other.m_itemMap )
   {}
   
   ~Private() {}
   
   void gcMark( uint32 mark )
   {
      ItemMap::iterator iter = m_itemMap.begin();
      while( iter != m_itemMap.end() )
      {
         iter->first.gcMark( mark );
         iter->second.gcMark( mark );
         ++iter;
      }
   }
};


ItemDict::ItemDict():
   _p( new Private ),
   m_flags(0),
   m_currentMark(0),
   m_version(0)
{}


ItemDict::ItemDict( const ItemDict& other ):
   _p( new Private(*other._p) ),
   m_flags( other.m_flags ),
   m_currentMark( other.m_currentMark ),
   m_version(0)
{}


ItemDict::~ItemDict()
{
   delete _p;
}


void ItemDict::gcMark( uint32 mark )
{
   if( m_currentMark == mark )
   {
      return;
   }

   m_currentMark = mark;
   _p->gcMark( mark );
}


void ItemDict::insert( const Item& key, const Item& value )
{     
   Item ckey;

   if ( (key.isString() && ! key.asString()->isImmutable()) || (key.isUser() && key.asClass()->isFlatInstance() ) )
   {
      if( ! key.clone(ckey) )
      {
         ckey.copyFromRemote(key);
      }
   }
   else
   {
      ckey.copyFromRemote(key);
   }

   _p->m_itemMap[ckey].copyFromRemote(value);
}



void ItemDict::merge( const ItemDict& other )
{
   Private* op = other._p;
   {
      Private::ItemMap::iterator iter = op->m_itemMap.begin();
      while( iter != op->m_itemMap.end() )
      {
         _p->m_itemMap[iter->first].copyFromRemote( iter->second );
         ++iter;
      }
   }

}


void ItemDict::clear()
{
   _p->m_itemMap.clear();
}


bool ItemDict::remove( const Item& key )
{
   Item keycp;
   keycp.copyFromRemote(key);
   bool result = _p->m_itemMap.erase(keycp);
   return result;
}


Item* ItemDict::find( const Item& key )
{
   Item keycp;
   keycp.copyFromRemote(key);
   Private::ItemMap::iterator iter = _p->m_itemMap.find(keycp);

   if( iter != _p->m_itemMap.end() )
   {
      return &iter->second;
   }

   return 0;                       
}



Item* ItemDict::find( const String& key )
{
   static Class* clsString = Engine::instance()->stdHandlers()->stringClass();
   Item keycp;

   keycp.setUser( clsString, &key );

   Private::ItemMap::iterator iter = _p->m_itemMap.find(keycp);

   if( iter != _p->m_itemMap.end() )
   {
      return &iter->second;
   }

   return 0;
}


length_t ItemDict::size() const
{
   length_t count = _p->m_itemMap.size();
   
   return count;
}


void ItemDict::describe( String& target, int depth, int maxlen ) const
{
   String ks, vs;
   
   if( size() == 0 )
   {
      target = "[=>]";
      return;
   }

   if( depth == 0 )
   {
      target = "...";
      return;
   }
   
   target.size(0);
   target += "[";

   Private::ItemMap::iterator kiter = _p->m_itemMap.begin();
   while( kiter != _p->m_itemMap.end() )
   {
      const Item& key = kiter->first;
      const Item& value = kiter->second;

      key.describe(ks,depth-1,maxlen);
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

Class* ItemDict::handler()
{
   static Class* handler = Engine::handlers()->dictClass();
   return handler;
}

void ItemDict::enumerate( Enumerator& rator ) const
{
   Private::ItemMap::iterator kiter = _p->m_itemMap.begin();
   while( kiter != _p->m_itemMap.end() )
   {
      rator( kiter->first, kiter->second );
      ++kiter;
   }
}

//========================================================
// Iterator class
//

class ItemDict::Iterator::Private
{
public:
   ItemArray m_pair;
   
   ItemDict::Private::ItemMap::const_iterator t_iter;
   
   Private() {
      m_pair.resize(2);
   }
   ~Private() {};
};



ItemDict::Iterator::Iterator( ItemDict* item ):
   GenericData( "ItemDict::Iterator" ),
   _pm( new Private ),
   m_dict( item ),
   m_version( item->version() ),
   m_currentMark(0),
   m_complete( false ),
   m_state( e_st_none )
{
   _pm->t_iter = item->_p->m_itemMap.begin();
}


ItemDict::Iterator::~Iterator()
{
   delete _pm;
}


void ItemDict::Iterator::gcMark( uint32 value )
{
   if( m_currentMark != value )
   {
      m_currentMark = value;
      _pm->m_pair.gcMark( value );
      if( m_dict != 0 )
      {
         m_dict->gcMark( value );
      }
   }
}


bool ItemDict::Iterator::gcCheck( uint32 value )
{
   // If all our components are old...
   if( _pm->m_pair.currentMark() < value
      && m_tempString.currentMark() < value
      && m_currentMark < value 
      )
   {
      return false; // item dead
   }
   
   return true;
}


ItemDict::Iterator* ItemDict::Iterator::clone() const
{
   if( m_dict == 0 ) return 0;
   return new Iterator( m_dict );
}


void ItemDict::Iterator::describe( String& target ) const
{
   target = "ItemDict::Iterator";
   if( m_dict == 0 || m_version != m_dict->version() )
   {
      target += " (invalidated)";
   }
}


bool ItemDict::Iterator::next( Item& target )
{
   static Class* ac = Engine::handlers()->arrayClass();
   
   if( m_dict == 0 ) return false;
   if( m_version != m_dict->version() )
   {
      m_dict = 0;
      return false;
   }

   if( m_dict->size() == 0 )
   {
      target.setBreak();
      return true;
   }
      
   if( m_complete )
   {
      target.setBreak(); 
      return true;
   }
    
   advance();
    
   // create a copied item, and ask to mark it for gc.
   target.copyFromLocal( Item(ac, &_pm->m_pair ) );
   if( ! m_complete )
   {
      target.setDoubt();
   }
   
   return true;
}


void ItemDict::Iterator::advance()
{
   
   fassert( m_dict != 0 );
   fassert( m_dict->version() == m_version );
   
   if( m_complete )
   {
      return;
   }

   ItemDict::Private* p = m_dict->_p;
   _pm->m_pair[0].lock();
   _pm->m_pair[0] = _pm->t_iter->first;
   _pm->m_pair[0].unlock();
   _pm->m_pair[1].lock();
   _pm->m_pair[1] = _pm->t_iter->second;
   _pm->m_pair[1].unlock();
   ++_pm->t_iter;

   if( _pm->t_iter == p->m_itemMap.end() )
   {
      m_complete = true;
   }
}


}

/* end of itemdict.cpp */
