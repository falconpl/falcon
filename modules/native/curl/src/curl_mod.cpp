/*
   FALCON - The Falcon Programming Language.
   FILE: curl_ext.cpp

   cURL library binding for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 27 Nov 2009 16:31:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: The above AUTHOR

         Licensed under the Falcon Programming Language License,
      Version 1.1 (the "License"); you may not use this file
      except in compliance with the License. You may obtain
      a copy of the License at

         http://www.falconpl.org/?page_id=license_1_1

      Unless required by applicable law or agreed to in writing,
      software distributed under the License is distributed on
      an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
      KIND, either express or implied. See the License for the
      specific language governing permissions and limitations
      under the License.

*/

/** \file
   cURL library binding for Falcon
   Internal logic functions - implementation.
*/

#include "curl_mod.h"
#include <falcon/stream.h>
#include <falcon/vm.h>
#include <falcon/coreslot.h>
#include <falcon/vmmsg.h>
#include <falcon/membuf.h>
#include <falcon/autocstring.h>

#include <stdio.h>
#include <string.h>

namespace Falcon {
namespace Mod {

CurlHandle::CurlHandle( const CoreClass* cls, bool bDeser ):
   CacheObject( cls, bDeser ),
   m_sReceived(0),
   m_dataStream(0),
   m_cbMode( e_cbmode_stdout ),
   m_readStream(0),
   m_pPostBuffer(0)
{
   if ( bDeser )
      m_handle = 0;
   else
   {
      m_handle = curl_easy_init();
      if (m_handle)
         curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_stdout );
   }
}

CurlHandle::CurlHandle( const CurlHandle &other ):
   CacheObject( other ),
   m_iDataCallback( other.m_iDataCallback ),
   m_sReceived(0),
   m_dataStream( other.m_dataStream ),
   m_sSlot( other.m_sSlot ),
   m_cbMode( e_cbmode_stdout )
{
   if ( other.m_handle != 0 )
      m_handle = curl_easy_duphandle( other.m_handle );
   else
      m_handle = 0;
}

CurlHandle::~CurlHandle()
{
   cleanup();
}


CurlHandle* CurlHandle::clone() const
{
   return new CurlHandle( *this );
}

void CurlHandle::cleanup()
{
   if( m_handle != 0 )
   {
      curl_easy_cleanup( m_handle );
      m_handle = 0;

      ListElement* head = m_slists.begin();
      while( head != 0 )
      {
         struct curl_slist* slist = (struct curl_slist*) head->data();
         curl_slist_free_all( slist );
         head = head->next();
      }
   }

   if (m_pPostBuffer != 0 )
   {
      memFree( m_pPostBuffer );
      m_pPostBuffer = 0;
   }
}

bool CurlHandle::serialize( Stream *stream, bool bLive ) const
{
   if ( ! bLive )
   {
      return false;
   }

   uint64 ptr = endianInt64( (uint64) m_handle );
   stream->write( &ptr, sizeof(ptr) );

   return CacheObject::serialize( stream, bLive );
}

bool CurlHandle::deserialize( Stream *stream, bool bLive )
{
   if ( ! bLive )
      return false;

   fassert( m_handle == 0 );

   uint64 ptr;
   if( stream->read( &ptr, sizeof(ptr) ) != sizeof(ptr) )
   {
      return false;
   }

   ptr = endianInt64( ptr );
   m_handle = (CURL*) ptr;

   return true;
}

void CurlHandle::gcMark( uint32 mark )
{
   memPool->markItem( m_iDataCallback );
   memPool->markItem( m_iReadCallback );

   if( m_sReceived != 0 )
      m_sReceived->mark( mark );

   if( m_dataStream != 0 )
      m_dataStream->gcMark( mark );

   if( m_readStream != 0 )
      m_readStream->gcMark( mark );

   CacheObject::gcMark( mark );
}

size_t CurlHandle::write_stdout( void *ptr, size_t size, size_t nmemb, void *)
{
   return fwrite( ptr, size, nmemb, stdout );
}


size_t CurlHandle::write_stream( void *ptr, size_t size, size_t nmemb, void *data)
{
   Stream* s = (Stream*) data;
   return s->write( ptr, size * nmemb );
}

size_t CurlHandle::write_msg( void *ptr, size_t size, size_t nmemb, void *data)
{
   VMachine* vm = VMachine::getCurrent();

   if( vm != 0 )
   {
      printf( "Received... %ld\n", size * nmemb );
      CurlHandle* cs = (CurlHandle*) data;
      VMMessage* vmmsg = new VMMessage( cs->m_sSlot );
      vmmsg->addParam( cs );
      CoreString* str = new CoreString;
      str->adopt( (char*) ptr, (int32) size * nmemb, 0 );
      str->bufferize();
      vmmsg->addParam( str );
      vm->postMessage( vmmsg );
   }

   return size * nmemb;
}

size_t CurlHandle::write_string( void *ptr, size_t size, size_t nmemb, void *data)
{
   CurlHandle* h = (CurlHandle*) data;
   if ( h->m_sReceived == 0 )
      h->m_sReceived = new CoreString( size * nmemb );

   String str;
   str.adopt( (char*) ptr, (int32) size * nmemb, 0 );
   h->m_sReceived->append( str );
   return size * nmemb;
}

size_t CurlHandle::write_callback( void *ptr, size_t size, size_t nmemb, void *data)
{
   VMachine* vm = VMachine::getCurrent();
   if( vm != 0 )
   {
      CurlHandle* self = (CurlHandle*) data;
      CoreString* str = new CoreString;
      str->adopt( ( char*) ptr, (int32) size * nmemb, 0 );
      vm->pushParameter( str );
      vm->callItemAtomic( self->m_iDataCallback, 1 );

      if( vm->regA().isNil() || (vm->regA().isBoolean() && vm->regA().asBoolean() ) )
         return size * nmemb;
      else if( vm->regA().isOrdinal() )
         return (size_t) vm->regA().forceInteger();
   }

   return 0;
}


void CurlHandle::setOnDataCallback( const Item& itm )
{
   m_sReceived = 0;
   m_dataStream = 0;

   m_iDataCallback = itm;
   m_cbMode = e_cbmode_callback;

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_callback );
      curl_easy_setopt( m_handle, CURLOPT_WRITEDATA, this );
   }

}


void CurlHandle::setOnDataStream( Stream* s )
{
   m_iDataCallback.setNil();
   m_sReceived = 0;

   m_dataStream = s;
   m_cbMode = e_cbmode_stream;

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_stream );
      curl_easy_setopt( m_handle, CURLOPT_WRITEDATA, s );
   }
}


void CurlHandle::setOnDataMessage( const String& msg )
{
   m_sReceived = 0;
   m_iDataCallback.setNil();
   m_dataStream = 0;

   m_sSlot = msg;
   m_cbMode = e_cbmode_slot;

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_msg );
      curl_easy_setopt( m_handle, CURLOPT_WRITEDATA, this );
   }

}


void CurlHandle::setOnDataGetString()
{
   // the string is initialized by the callback
   m_sReceived = 0;
   m_iDataCallback.setNil();
   m_dataStream = 0;

   m_cbMode = e_cbmode_string;

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_string );
      curl_easy_setopt( m_handle, CURLOPT_WRITEDATA, this );
   }
}


void CurlHandle::setOnDataStdOut()
{
   // the string is initialized by the callback
   m_sReceived = 0;
   m_iDataCallback.setNil();
   m_dataStream = 0;

   m_cbMode = e_cbmode_stdout;
   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_stdout );
   }
}

void CurlHandle::setReadCallback( const Item& callable )
{
   // the string is initialized by the callback
   m_iReadCallback = callable;
   m_readStream = 0;

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_READFUNCTION, read_callback );
      curl_easy_setopt( m_handle, CURLOPT_READDATA, this );
   }
}

void CurlHandle::setReadStream( Stream* stream )
{
   // the string is initialized by the callback
   m_iReadCallback.setNil();
   m_readStream = stream;

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_READFUNCTION, read_stream );
      curl_easy_setopt( m_handle, CURLOPT_READDATA, this );
   }
}


CoreString* CurlHandle::getData()
{
   CoreString* ret = m_sReceived;
   m_sReceived = 0;
   return ret;
}

size_t CurlHandle::read_callback( void *ptr, size_t size, size_t nmemb, void *data)
{
   VMachine* vm = VMachine::getCurrent();
   if ( vm != 0 )
   {
      CurlHandle* h = (CurlHandle *) data;
      MemBuf_1 m( (byte*) ptr, size* nmemb, 0 );
      vm->pushParameter( (MemBuf*) &m );
      vm->callItemAtomic( h->m_iReadCallback, 1 );

      if( vm->regA().isOrdinal() )
         return (size_t) vm->regA().forceInteger();

   }

   return 0;
}


size_t CurlHandle::read_stream( void *ptr, size_t size, size_t nmemb, void *data)
{
   CurlHandle* h = (CurlHandle *) data;
   if( h->m_readStream != 0 )
   {
      return h->m_readStream->read( ptr, size * nmemb );
   }

   return CURL_READFUNC_ABORT;
}

struct curl_slist* CurlHandle::slistFromArray( CoreArray* ca )
{
   struct curl_slist* sl = NULL;

   for( uint32 pos = 0; pos < ca->length(); ++pos )
   {
      Item& current = ca->at(pos);
      if( ! current.isString() )
      {
         if( sl != 0 )
            m_slists.pushBack( sl );
         return 0;
      }

      AutoCString str( current );
      sl = curl_slist_append( sl, str.c_str() );
   }

   if( sl != 0 )
      m_slists.pushBack( sl );

   return sl;
}

void CurlHandle::postData( const String& str )
{
   if (m_pPostBuffer != 0 )
      memFree( m_pPostBuffer );

   m_pPostBuffer = memAlloc( str.size() );
   memcpy(m_pPostBuffer, str.getRawStorage(), str.size() );

   curl_easy_setopt( handle(), CURLOPT_POSTFIELDS, m_pPostBuffer );
   curl_easy_setopt( handle(), CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t) str.size() );
}




CoreObject* CurlHandle::Factory( const CoreClass *cls, void *data, bool deser )
{
   return new CurlHandle( cls, deser );
}




//==============================================================
//

CurlMultiHandle::CurlMultiHandle( const CoreClass* cls, bool bDeser ):
   CacheObject( cls, bDeser )
{
   if ( bDeser )
      m_handle = 0;
   else
   {
      m_handle = curl_multi_init();
      m_mtx = new Mutex;
      m_refCount = new int(1);
   }
}


CurlMultiHandle::CurlMultiHandle( const CurlMultiHandle &other ):
   CacheObject( other )
{
   if( other.m_handle != 0 )
   {
      m_mtx = other.m_mtx;
      m_refCount = other.m_refCount;
      m_handle = other.m_handle;

      m_mtx->lock();
      (*m_refCount)++;
      m_mtx->unlock();
   }
   else
   {
      m_mtx = new Mutex;
      m_refCount = 0;
   }

}

CurlMultiHandle::~CurlMultiHandle()
{
   if ( m_handle != 0 )
   {
      m_mtx->lock();
      bool bDelete = --(*m_refCount) == 0;
      m_mtx->unlock();

      if( bDelete )
      {
         delete m_refCount;
         delete m_mtx;
         curl_multi_cleanup( m_handle );
      }
   }
}

CurlMultiHandle* CurlMultiHandle::clone() const
{
   return new CurlMultiHandle( *this );
}

bool CurlMultiHandle::serialize( Stream *stream, bool bLive ) const
{
   if ( ! bLive )
      return false;

   // incref immediately
   m_mtx->lock();
   (*m_refCount)++;
   m_mtx->unlock();

   uint64 ptrh = endianInt64( (uint64) m_handle );
   uint64 ptrm = endianInt64( (uint64) m_mtx );
   uint64 ptrrc = endianInt64( (uint64) m_refCount );
   stream->write( &ptrh, sizeof(ptrh) );
   stream->write( &ptrm, sizeof(ptrm) );
   stream->write( &ptrrc, sizeof(ptrrc) );

   bool bOk = CacheObject::serialize( stream, bLive );

   if( ! bOk )
   {
      m_mtx->lock();
      (*m_refCount)--;
      m_mtx->unlock();
   }

   return true;
}

bool CurlMultiHandle::deserialize( Stream *stream, bool bLive )
{
   if ( ! bLive )
      return false;

   fassert( m_handle == 0 );

   uint64 ptrh;
   uint64 ptrm;
   uint64 ptrrc;

   if( stream->read( &ptrh, sizeof(ptrh) ) != sizeof( ptrh )  ||
       stream->read( &ptrm, sizeof(ptrm) ) != sizeof( ptrm )  ||
       stream->read( &ptrrc, sizeof(ptrrc) ) != sizeof( ptrrc )  )
   {
      return false;
   }

   m_handle = (CURLM*) endianInt64( ptrh );
   m_mtx = (Mutex*) endianInt64( ptrm );
   m_refCount = (int*) endianInt64( ptrrc );

   return true;
}

CoreObject* CurlMultiHandle::Factory( const CoreClass *cls, void *data, bool bDeser )
{
   return new CurlMultiHandle( cls, bDeser );
}


void CurlMultiHandle::gcMark( uint32 mark )
{
   m_handles.gcMark( mark );
   CacheObject::gcMark( mark );
}


bool CurlMultiHandle::addHandle( CurlHandle* h )
{
   for ( uint32 i = 0; i < m_handles.length(); ++i )
   {
      if ( m_handles[i].asObjectSafe() == h )
         return false;
   }

   m_handles.append( h );
   curl_multi_add_handle( handle(), h->handle() );
   return true;
}


bool CurlMultiHandle::removeHandle( CurlHandle* h )
{
   for ( uint32 i = 0; i < m_handles.length(); ++i )
   {
      if ( m_handles[i].asObjectSafe() == h )
      {
         curl_multi_remove_handle( handle(), h->handle() );
         m_handles.remove( i );
         return true;
      }
   }

   return false;
}


}
}


/* end of curl_mod.cpp */
