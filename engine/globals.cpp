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


namespace Falcon
{
#define FLC_DECLARE_ENGINE_MSG
#include <falcon/eng_messages.h>
#undef FLC_DECLARE_ENGINE_MSG

StringTable * engineStrings;

namespace Engine
{
   static Mutex s_mtx;
   static Map *s_serviceMap = 0;
   static String* s_sIOEnc = 0;
   static String* s_sSrcEnc = 0;

   bool addVFS( const String &name, VFSProvider *prv )
   {
      s_mtx.lock();
      if ( s_serviceMap == 0 )
      {
         s_serviceMap = new Map( &traits::t_string(), &traits::t_voidp() );
      }
      else
      {
         void *oldPrv = s_serviceMap->find( &name );
         if( oldPrv != 0 )
         {
            s_mtx.unlock();
            return false;
         }
      }

      s_serviceMap->insert( &name, prv );
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

   void Shutdown()
   {

      delete memPool;
      memPool = 0;

      delete engineStrings;
      engineStrings = 0;
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
         return true;
      }
      // ... signal that we didn't found the language.
      return false;
   }

   void setEncodings( const String &sSrcEnc, const String &sIOEnc )
   {
      s_mtx.lock();
      delete s_sSrcEnc;
      delete s_sIOEnc;
      s_sSrcEnc = new String(sSrcEnc);
      s_sIOEnc = new String(sIOEnc);
      s_sSrcEnc->bufferize();
      s_sIOEnc->bufferize();
      s_mtx.unlock();
   }

   void getEncodings( String &sSrcEnc, String &sIOEnc )
   {
      s_mtx.lock();
      sSrcEnc = *s_sSrcEnc;
      sIOEnc = *s_sIOEnc;
      s_mtx.unlock();
   }
}

}


/* end of globals.cpp */
