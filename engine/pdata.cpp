/*
   FALCON - The Falcon Programming Language.
   FILE: pdata.cpp

   Engine, VM Process or processor specific persistent data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 13 Jan 2014 16:22:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/pdata.cpp"

#include <falcon/trace.h>
#include <falcon/pdata.h>
#include <falcon/item.h>
#include <falcon/gclock.h>
#include <falcon/mt.h>
#include <falcon/string.h>

#include <map>

namespace Falcon  {

class PData::Private
{
public:
   typedef std::map<String, GCLock*> PDataMap;
   PDataMap m_pdata;
   Mutex m_mtx;

   Private()
   {
   }

   ~Private()
   {
     PDataMap::iterator iter = m_pdata.begin();
     while( iter != m_pdata.end() )
     {
        GCLock* gl = iter->second;
        gl->dispose();
        ++iter;
     }
  }

};


PData::PData()
{
   TRACE2( "PData::PData -- Created PDATA object %p", this );
   _p = new Private;
}

PData::~PData()
{
   TRACE2( "PData::~PData -- Destroyed PDATA object %p", this );
   delete _p;
}


bool PData::set( const String& id, const Item& data ) const
{
   TRACE2( "PData::set object %s as %s",
            id.c_ize(), data.describe().c_ize() );

   // get the thread-specific data map
   Private::PDataMap* pm = &_p->m_pdata;

   // search the key
   _p->m_mtx.lock();
   Private::PDataMap::iterator iter = pm->find( id );

   // already around?
   if( iter != pm->end() )
   {
      GCLock* gl = iter->second;
      gl->item() = data;
      _p->m_mtx.unlock();

      return false;
   }
   else
   {
      (*pm)[id] = Engine::collector()->lock( data );
   }
   _p->m_mtx.unlock();
   return true;
}

bool PData::get( const String& id, Item& data, bool bPlaceHolder ) const
{
   TRACE2( "PData::get object %s as %s (%s)",
               id.c_ize(), data.describe().c_ize(), (bPlaceHolder ? "placeholder" : "no placeholder") );

   // get the thread-specific data map
   Private::PDataMap* pm = &_p->m_pdata;


   // search the key
   _p->m_mtx.lock();
   Private::PDataMap::iterator iter = pm->find( id );

   // already around?
   if( iter != pm->end() )
   {
      GCLock* gl = iter->second;
      data = gl->item();
      _p->m_mtx.unlock();
      return true;
   }
   else if ( bPlaceHolder )
   {
      (*pm)[id] = Engine::collector()->lock( Item() );
   }

   // didn't find it
   _p->m_mtx.unlock();
   return false;
}


bool PData::remove( const String& id ) const
{
   TRACE2( "PData::remove object %s",
               id.c_ize() );

   // get the thread-specific data map
   Private::PDataMap* pm = &_p->m_pdata;

   // search the key
   _p->m_mtx.lock();
   Private::PDataMap::iterator iter = pm->find( id );

   // already around?
   if( iter != pm->end() )
   {
      GCLock* gl = iter->second;
      pm->erase(iter);
      _p->m_mtx.unlock();

      gl->dispose();
      return true;
   }
   else {
      _p->m_mtx.unlock();
      return false;
   }
}


}

/* end of pdata.cpp */


