/*
   FALCON - The Falcon Programming Language.
   FILE: modulecache.h

   Cache-sensible module loader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 30 May 2010 15:12:30 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_MODULECACHE_H_
#define FALCON_MODULECACHE_H_

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/string.h>
#include <falcon/module.h>
#include <falcon/mt.h>
#include <falcon/genericmap.h>

namespace Falcon {

/** The cache where modules are stored.

    Updates are threadsafe.
 */
class ModuleCache: public BaseAlloc
{
public:
   ModuleCache();

   /** When the cache is destroyed, all the modules in it are decreffed. */
   ~ModuleCache();

   /** Adds a module to the cache.
      The module is increffed when added to the cache. If another module with the
      same name is already in the map, the old module is returned (increffed),
      and the incoming module is decreffed.
   */
   Module* add( const String& muri, Module* module );

   /** Removes a module from the cache.
       If the module is in the cache, it is decreffed.
   */
   bool remove( const String& muri );

   /** Returns a module if it is in cache, or 0 if not found.
       The returned instance is increffed.
   */
   Module* find( const String& muri );

private:
   Mutex m_mtx;
   int m_refCount;
   Map m_modMap;
};

}

#endif

/* end of modulecache.h */
