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

namespace Falcon {

ModuleCache::ModuleCache():
      m_modMap( &traits::t_string(), &traits::t_voidp() )
{}

ModuleCache::~ModuleCache()
{
   MapIterator iter = m_modMap.begin();
   while( iter.hasCurrent() )
   {
      Module* mod = *(Module**) iter.currentValue();
      mod->decref();
      iter.next();
   }
}

Module* ModuleCache::add( const String& muri, Module* module )
{
   m_mtx.lock();
   void* data = m_modMap.find( &muri );
   if( data != 0 )
   {
      Module* mod1 = *(Module**) data;
      mod1->incref();
      m_mtx.unlock();

      module->decref();
      return mod1;
   }
   else
   {
      m_modMap.insert( &muri, module );
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
      Module* mod = *(Module**) iter.currentValue();
      m_modMap.erase( iter );
      m_mtx.unlock();

      mod->decref();
      return true;
   }

   m_mtx.unlock();
   return false;
}

Module* ModuleCache::find( const String& muri )
{
   m_mtx.lock();
   void* data = m_modMap.find( &muri );
   if( data != 0 )
   {
      Module* mod = *(Module**) data;
      mod->incref();
      m_mtx.unlock();

      return mod;
   }

   m_mtx.unlock();
   return 0;
}

}

/* end of modulecache.cpp */
