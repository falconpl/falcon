/*
   FALCON - The Falcon Programming Language
   FILE: globals.cpp

   Engine static/global data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 02 Mar 2009 20:33:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Engine static/global data setup and initialization
*/

#include <falcon/globals.h>
#include <falcon/genericmap.h>
#include <falcon/string.h>
#include <falcon/mt.h>
#include <falcon/mempool.h>
#include <falcon/vfs_file.h>
#include <falcon/strtable.h>
#include <falcon/modulecache.h>

namespace Falcon
{
#define FLC_DECLARE_ENGINE_MSG
#include <falcon/eng_messages.h>
#undef FLC_DECLARE_ENGINE_MSG

StringTable * engineStrings = 0;

namespace Engine
{
   static Mutex s_mtx;
   static Map *s_serviceMap = 0;
   static String* s_sIOEnc = 0;
   static String* s_sSrcEnc = 0;
   static String* s_searchPath = 0;
   static ModuleCache* s_moduleCache = 0;

#ifdef FALCON_SYSTEM_WIN
   static bool s_bWindowsNamesConversion = true;
#else
   static bool s_bWindowsNamesConversion = false;
#endif

   /** Release language data. */
   void releaseLanguage();
   /** Release encoding data. */
   void releaseEncodings();

   bool addVFS( VFSProvider *prv )
   {
      fassert( prv != 0 );
      return addVFS(prv->protocol(),prv);
   }

   bool addVFS( const String &protocol, VFSProvider *prv )
   {
      fassert( prv != 0 );

      s_mtx.lock();
      if ( s_serviceMap == 0 )
      {
         s_serviceMap = new Map( &traits::t_string(), &traits::t_voidp() );
      }
      else
      {
         void *oldPrv = s_serviceMap->find( &protocol );
         if( oldPrv != 0 )
         {
            s_mtx.unlock();
            return false;
         }
      }

      s_serviceMap->insert( &protocol, prv );
      s_mtx.unlock();

      return true;
   }

   VFSProvider* getVFS( const String &name )
   {
      s_mtx.lock();
      if ( s_serviceMap == 0 )
      {
         s_mtx.unlock();
         return 0;
      }

      void *prv = s_serviceMap->find( &name );
      s_mtx.unlock();
      if ( prv != 0 )
         return *(VFSProvider**) prv;

      return 0;
   }

   void Init()
   {
      // Default language
      setLanguage( "C" );
      setEncodings( "C", "C" );

      // create the default mempool.
      memPool = new MemPool;
      memPool->start();

      // create the default file VSF
      addVFS( "file", new VFSFile );
      addVFS( "", new VFSFile );

      // Ok, we're ready
   }

   void PerformGC()
   {
     memPool->performGC();
   }
   
   void Shutdown()
   {
      if( s_searchPath != 0 )
         delete s_searchPath;
     
      delete memPool;
      memPool = 0;

      delete s_moduleCache;
      s_moduleCache = 0;

      releaseLanguage();
      releaseEncodings();

      // clear all the service ( and VSF );
      s_mtx.lock();
      if( s_serviceMap )
      {
         MapIterator mi = s_serviceMap->begin();
         while ( mi.hasCurrent() ) {
            delete *(VFSProvider**) mi.currentValue();
            mi.next();
         }

         delete s_serviceMap;
         s_serviceMap = 0;
      }
      s_mtx.unlock();

	  traits::releaseTraits();

	  gcMemShutdown(); 
   }



   const String &getMessage( uint32 id )
   {
      const String *data = engineStrings->get( id );
      fassert( data != 0 );
      return *data;
   }

   bool setTable( StringTable *tab )
   {
      if ( engineStrings == 0 || engineStrings->size() == tab->size() )
      {
         engineStrings = tab;
         return true;
      }
      return false;
   }

   bool setLanguage( const String &language )
   {
      delete engineStrings;
      engineStrings = new StringTable;
      if( language == "C" )
      {
         #define  FLC_REALIZE_ENGINE_MSG
         #include <falcon/eng_messages.h>
         #undef FLC_REALIZE_ENGINE_MSG
         return true;
      }
      // ... signal that we didn't found the language.
      return false;
   }

   void releaseLanguage()
   {
	   delete engineStrings;
	   engineStrings = 0;
   }

   void setEncodings( const String &sSrcEnc, const String &sIOEnc )
   {
      s_mtx.lock();
      
      releaseEncodings();

      s_sSrcEnc = new String(sSrcEnc);
      s_sIOEnc = new String(sIOEnc);
      s_sSrcEnc->bufferize();
      s_sIOEnc->bufferize();
      s_mtx.unlock();
   }

   void releaseEncodings()
   {
	   delete s_sSrcEnc;
	   delete s_sIOEnc;
   }

   void getEncodings( String &sSrcEnc, String &sIOEnc )
   {
      s_mtx.lock();
      sSrcEnc = *s_sSrcEnc;
      sIOEnc = *s_sIOEnc;
      s_mtx.unlock();
   }

   void setSearchPath( const String &str )
   {
      s_mtx.lock();
      if( s_searchPath != 0 )
         delete s_searchPath;

      s_searchPath = new String( str );
      s_searchPath->bufferize();
      s_mtx.unlock();
   }

   String getSearchPath()
   {
      s_mtx.lock();
      if( s_searchPath != 0 )
      {
         String temp( *s_searchPath );
         temp.bufferize();
         s_mtx.unlock();
         return temp;
      }

      s_mtx.unlock();
      return "";
   }

   void setWindowsNamesConversion( bool s )
   {
      s_bWindowsNamesConversion = s;
   }

   bool getWindowsNamesConversion()
   {
      return s_bWindowsNamesConversion;
   }

   void cacheModules( bool tmode )
   {
      s_mtx.lock();
      if ( tmode )
      {
         if ( s_moduleCache == 0 )
            s_moduleCache = new ModuleCache;
      }
      else
      {
         delete s_moduleCache;
         s_moduleCache = 0;
      }
      s_mtx.unlock();
   }

   ModuleCache* getModuleCache()
   {
      return s_moduleCache;
   }
}

}


/* end of globals.cpp */
