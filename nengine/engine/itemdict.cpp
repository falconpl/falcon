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

#include <map>

namespace Falcon
{

class ItemDict::Private
{
public:
   typedef std::map<Item, Item> ItemMap;
   ItemMap m_items;

   Private() {}
   ~Private() {}
};


ItemDict::ItemDict():
   _p( new Private ),
   m_flags(0),
   m_currentMark(0)
{

}


ItemDict::ItemDict( const ItemDict& other ):
   _p( new Private ),
   m_flags( other.m_flags ),
   m_currentMark( other.m_currentMark )
{
   _p->m_items = other._p->m_items;
}


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
   
   Private::ItemMap& dict = _p->m_items;
   Private::ItemMap::iterator pos = dict.begin();

   while( pos != dict.end() )
   {
      const Item& key = pos->first;
      const Item& value = pos->second;
      if( key.isUser() && key.isGarbaged() )
      {
         key.asClass()->gcMark(key.asInst(), mark);
      }

      if( value.isUser() && value.isGarbaged() )
      {
         value.asClass()->gcMark(value.asInst(), mark);
      }

      ++pos;
   }
}


void ItemDict::insert( const Item& key, const Item& value )
{
   _p->m_items[key] = value;
}


void ItemDict::remove( const Item& key )
{
   _p->m_items.erase( key );
}


Item* ItemDict::find( const Item& key )
{
   Private::ItemMap& dict = _p->m_items;
   Private::ItemMap::iterator pos = dict.find( key );
   if( pos == dict.end() )
   {
      return 0;
   }

   return &pos->second;
}


length_t ItemDict::size() const
{
   return _p->m_items.size();
}


void ItemDict::describe( String& target, int depth, int maxlen ) const
{
   if( depth == 0 )
   {
      target = "...";
      return;
   }
   
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
}


void ItemDict::enumerate( Enumerator& rator )
{
   Private::ItemMap& dict = _p->m_items;
   Private::ItemMap::iterator pos = dict.begin();

   while( pos != dict.end() )
   {
      rator( pos->first, pos->second );
      ++pos;
   }
}


}

/* end of itemdict.cpp */
