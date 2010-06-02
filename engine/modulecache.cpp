/*
   FALCON - The Falcon Programming Language.
   FILE: modulecache.cpp

   Cache-sensible module loader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 30 May 2010 15:12:30 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/modulecache.h>
#include <falcon/traits.h>

#include <falcon/basealloc.h>
#include <falcon/timestamp.h>
#include <falcon/sys.h>


namespace Falcon {

class CacheEntry: public BaseAlloc
{
public:
   CacheEntry( Module* mod, const TimeStamp& tsDate ):
      m_module( mod ),
      m_ts( tsDate )
      {}

   ~CacheEntry()
   {
      m_module->decref();
   }

   void change( Module* mod, const TimeStamp& tsDate )
   {
      m_module->decref();
      mod->incref();
      m_module = mod;
      m_ts = tsDate;
   }

   Module* m_module;
   TimeStamp m_ts;
};

ModuleCache::ModuleCache():
      m_modMap( &traits::t_string(), &traits::t_voidp() )
{}

ModuleCache::~ModuleCache()
{
   MapIterator iter = m_modMap.begin();
   while( iter.hasCurrent() )
   {
      CacheEntry* mod = *(CacheEntry**) iter.currentValue();
      delete mod;
      iter.next();
   }
}

Module* ModuleCache::add( const String& muri, Module* module )
{
   FileStat fm;
   bool gotStats = Sys::fal_stats( muri, fm );

   m_mtx.lock();
   void* data = m_modMap.find( &muri );
   if( data != 0 )
   {
      CacheEntry* mod_cache = *(CacheEntry**) data;


      // New module?
      // -- if I can't get the stats, in doubt, change it
      if( ! gotStats || fm.m_mtime->compare( mod_cache->m_ts ) > 0 )
      {
         mod_cache->change( module, *fm.m_mtime );
         m_mtx.unlock();

         return module;
      }
      else
      {
         Module* mod1 = mod_cache->m_module;
         mod1->incref();
         m_mtx.unlock();

         module->decref();
         return mod1;
      }
   }
   else
   {
      m_modMap.insert( &muri, new CacheEntry( module, *fm.m_mtime ) );
      module->incref();
      m_mtx.unlock();
      return module;
   }
}

bool ModuleCache::remove( const String& muri )
{
   MapIterator iter;

   m_mtx.lock();
   if( m_modMap.find( &muri, iter ) )
   {
      CacheEntry* mod = *(CacheEntry**) iter.currentValue();
      m_modMap.erase( iter );
      m_mtx.unlock();

      delete mod;
      return true;
   }

   m_mtx.unlock();
   return false;
}

Module* ModuleCache::find( const String& muri )
{
   FileStat fm;
   bool gotStats = Sys::fal_stats( muri, fm );

   m_mtx.lock();
   void* data = m_modMap.find( &muri );
   if( data != 0 )
   {
      CacheEntry* emod = *(CacheEntry**) data;
      if ( !gotStats || fm.m_mtime->compare( emod->m_ts ) > 0 )
      {
         // ignore the find
         m_mtx.unlock();
         return 0;
      }

      Module* mod = emod->m_module;
      mod->incref();
      m_mtx.unlock();

      return mod;
   }

   m_mtx.unlock();
   return 0;
}

}

/* end of modulecache.cpp */
