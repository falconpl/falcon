/*
   FALCON - The Falcon Programming Language.
   FILE: delegatemap.cpp

   Falcon core module -- delegate map
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Apr 2013 21:06:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/delegatemap.cpp"

#include <falcon/item.h>
#include <falcon/delegatemap.h>

#include <map>

namespace Falcon {

class DelegateMap::Private
{
public:
   typedef std::map<String, Item> DMap;
   DMap m_dmap;

   Item m_general;
};


DelegateMap::DelegateMap():
   m_mark(0),
   m_bHasGeneral(false)
{
   _p = new Private;
}


DelegateMap::~DelegateMap()
{
   delete _p;
}

void DelegateMap::setDelegate( const String& msg, const Item& target )
{
   m_mtx.lock();
   if( msg == "*" )
   {
      m_bHasGeneral = true;
      _p->m_general = target;
   }
   else {
      _p->m_dmap[msg] = target;
   }
   m_mtx.unlock();
}

bool DelegateMap::getDelegate( const String& msg, Item& target ) const
{
   bool result = false;

   m_mtx.lock();
   if( m_bHasGeneral )
   {
      result = true;
      target = _p->m_general;
   }
   else {
      Private::DMap::iterator iter = _p->m_dmap.find(msg);
      if( iter != _p->m_dmap.end() )
      {
         target = iter->second;
         result = true;
      }
   }
   m_mtx.unlock();

   return result;
}


void DelegateMap::clear()
{
   m_mtx.lock();
   m_bHasGeneral = false;
   _p->m_dmap.clear();
   m_mtx.unlock();
}


void DelegateMap::gcMark(uint32 mark)
{
   if( m_mark != mark )
   {
      m_mark = mark;
      m_mtx.lock();
      if( m_bHasGeneral )
      {
         _p->m_general.gcMark(mark);
      }

      Private::DMap::iterator iter = _p->m_dmap.begin();
      while( iter != _p->m_dmap.end() )
      {
         iter->second.gcMark(mark);
         ++iter;
      }
      m_mtx.unlock();
   }
}

}

/* delegatemap.cpp */
