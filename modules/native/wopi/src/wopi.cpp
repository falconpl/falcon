/*
   FALCON - The Falcon Programming Language.
   FILE: wopi.cpp

   Falcon Web Oriented Programming Interface.

   Global WOPI application objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 25 Apr 2010 17:02:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/wopi.h>
#include <falcon/wopi/error_ext.h>
#include <falcon/vm.h>
#include <falcon/stringstream.h>
#include <falcon/module.h>
#include <falcon/attribmap.h>
#include <falcon/garbagelock.h>

namespace Falcon {
namespace WOPI {

Wopi::Wopi():
      m_pdata( Wopi::pdata_deletor )
{
}

Wopi::~Wopi()
{
   AppDataMap::iterator iter = m_admap.begin();
   while( iter != m_admap.end() )
   {
      delete iter->second;
      ++iter;
   }
}


void Wopi::pdata_deletor( void* data )
{
   // maybe an overkill, but...
   if ( data == 0 )
      return;

   PDataMap* pm = (PDataMap*) data;
   PDataMap::iterator iter = pm->begin();

   while( iter != pm->end() )
   {
      delete iter->second;
      ++iter;
   }

   delete pm;
}


void Wopi::dataLocation( const String& loc )
{
   m_mtx.lock();
   m_sAppDataLoc = loc;
   m_mtx.unlock();
}

String Wopi::dataLocation()
{
   m_mtx.lock();
   String ret = m_sAppDataLoc;
   m_mtx.unlock();

   return ret;
}


bool Wopi::setData( Item& data, const String& appName, bool atomicUpdate )
{
   SharedMem* pAppMem = 0;

   // do we have the required appname data?
   m_mtx.lock();
   AppDataMap::const_iterator pos = m_admap.find( appName );
   if( pos == m_admap.end() )
   {
      m_mtx.unlock();
      pAppMem = inner_create_appData( appName );
   }
   else
   {

      pAppMem = pos->second;
      m_mtx.unlock();
   }

   // we can deal with the shared memory in an unlocked region, of course.
   if ( pAppMem->currentVersion() != pAppMem->lastVersion() )
   {
      // we already know we're out of sync.
      if( atomicUpdate )
         inner_readData( pAppMem, data );

      return false;
   }

   // ok, try and serialize the data.
   StringStream source;
   Item::e_sercode sc = data.serialize( &source, false );
   if( sc != Item::sc_ok )
   {
      throw new WopiError( ErrorParam( FALCON_ERROR_WOPI_APPDATA_SER, __LINE__ )
                  .desc( "Error during Serialization of application data")
                  .extra( String("type ").N( (int) sc ) )
                  );
   }

   // great, the data is serialized; try to get it out of the door.
   int32 datalen = (int32) source.tell();
   source.seekBegin(0);
   bool bSuccess = pAppMem->commit( &source, datalen, atomicUpdate );

   // did we succeed?
   if( ! bSuccess )
   {
      // No; we wasted the serialization. However, are now required to
      // deserialize the item?
      if( atomicUpdate )
      {
         // ... and, is there anything to be de-serialized?
         if( source.tell() != 0 )
         {
            source.seekBegin(0);
            Item::e_sercode sc = data.deserialize( &source, VMachine::getCurrent() );
            if( sc != Item::sc_ok )
            {
               throw new WopiError( ErrorParam( FALCON_ERROR_WOPI_APPDATA_DESER, __LINE__ )
                     .desc( "Error during de-serialization of application data")
                     .extra( String("type ").N( (int) sc ) )
                     );
            }
         }
         else
         {
            data.setNil();
         }
      }
   }

   return bSuccess;
}


bool Wopi::getData( Item& data, const String& appName )
{
   SharedMem* shmem = 0;

   // do we have the required appname data?
   m_mtx.lock();
   AppDataMap::const_iterator pos = m_admap.find( appName );
   if( pos == m_admap.end() )
   {
      m_mtx.unlock();
      shmem = inner_create_appData( appName );
   }
   else
   {
      shmem = pos->second;
      m_mtx.unlock();
   }

   // we can deal with the shared memory in an unlocked region, of course.
   inner_readData( shmem, data );
   return true;
}

SharedMem* Wopi::inner_create_appData( const String& appName )
{
   SharedMem* pAppMem = 0;

   // a new application data. Maybe.
   String sAppName = appName == "" ? "DFLT_"
         : "N_" + appName;

   // shall we get a backed up application data?
   if( m_sAppDataLoc != "" )
   {
      String sAppLoc = appName == "" ? m_sAppDataLoc + "/_WOPI_DEFAULT_DATA"
                  : m_sAppDataLoc + "/" + appName + ".fdt";
      pAppMem = new SharedMem( sAppName, sAppLoc );
   }
   else
   {
      pAppMem = new SharedMem( sAppName );
   }

   // ok; but someone may have added the shared mem in the meanwhile.
   // If it's so, ok, np.  Just discard our copy.
   m_mtx.lock();
   AppDataMap::const_iterator pos = m_admap.find( appName );
   if( pos == m_admap.end() )
   {
      m_admap[ appName ] = pAppMem;
      m_mtx.unlock();
   }
   else
   {
      SharedMem* pOld = pAppMem;
      pAppMem = pos->second;
      m_mtx.unlock();

      delete pOld;
   }

   return pAppMem;
}


void Wopi::inner_readData( SharedMem* shmem, Item& data )
{
   StringStream target;
   shmem->read( &target, true );

   if ( target.tell() == 0 )
   {
      // nothing to be deserialized.
      data.setNil();
      return;
   }

   target.seekBegin(0);
   Item::e_sercode sc = data.deserialize( &target, VMachine::getCurrent() );
   if( sc != Item::sc_ok )
   {
      throw new WopiError( ErrorParam( FALCON_ERROR_WOPI_APPDATA_DESER, __LINE__ )
            .desc( "Error during de-serialization of application data")
            .extra( String("type ").N( (int) sc ) )
            );
   }
}


bool Wopi::setPersistent( const String& id, const Item& data )
{
   // get the thread-specific data map
   PDataMap* pm = (PDataMap*) m_pdata.get();

   // we don't have it?
   if( pm == 0 )
   {
      pm = new PDataMap;
      m_pdata.set( pm );
   }

   // search the key
   PDataMap::iterator iter = pm->find( id );

   // already around?
   if( iter != pm->end() )
   {
      GarbageLock* gl = iter->second;

      // if data is NIL, we can destroy the entry
      if( data.isNil() )
      {
         delete gl;
         pm->erase(iter);
      }
      else
      {
         gl->item() = data;
      }

      return false;
   }
   else if ( ! data.isNil() )
   {
      (*pm)[id] = new GarbageLock( data );
   }

   return true;
}


bool Wopi::getPeristent( const String& id, Item& data ) const
{
   // get the thread-specific data map
   PDataMap* pm = (PDataMap*) m_pdata.get();

   // we don't have it?
   if( pm == 0 )
   {
      // then we can hardly have the key
      return false;
   }

   // search the key
   PDataMap::iterator iter = pm->find( id );

   // already around?
   if( iter != pm->end() )
   {
      GarbageLock* gl = iter->second;
      data = gl->item();
      return true;
   }

   // didn't find it
   return false;
}


//========================================================================
// CoreWopi
//

CoreWopi::CoreWopi( const CoreClass* parent ):
   CoreObject( parent ),
   m_wopi( 0 )
{
}

CoreWopi::~CoreWopi()
{

}

CoreObject *CoreWopi::clone() const
{
   return 0;
}


bool CoreWopi::setProperty( const String &prop, const Item &value )
{
   readOnlyError( prop );
   return false;
}


bool CoreWopi::getProperty( const String &prop, Item &value ) const
{
   return defaultProperty( prop, value );
}


void CoreWopi::configFromModule( const Module* mod )
{
   AttribMap* attribs = mod->attributes();
   if( attribs == 0 )
   {
      return;
   }

   VarDef* value = attribs->findAttrib( FALCON_WOPI_PDATADIR_ATTRIB );
   if( value != 0 && value->isString() )
   {
      wopi()->dataLocation( *value->asString() );
   }
}

CoreObject* CoreWopi::factory( const CoreClass *cls, void *, bool )
{
   return new CoreWopi( cls );
}

}
}

/* end of wopi.cpp */
