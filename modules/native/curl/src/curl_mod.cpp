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
#include <falcon/autocstring.h>
#include <falcon/wvmcontext.h>

#include <stdio.h>
#include <string.h>

namespace Falcon {
namespace Mod {

CurlHandle::CurlHandle(const Class* maker):
   m_sReceived(0),
   m_dataStream(0),
   m_cbMode( e_cbmode_stdout ),
   m_readStream(0),
   m_pPostBuffer(0)
{
   m_handle = curl_easy_init();
   m_class = maker;

   m_context = 0;

   if (m_handle)
   {
      curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_stdout );
   }
}

CurlHandle::CurlHandle( const CurlHandle &other ):
   m_iDataCallback( other.m_iDataCallback ),
   m_sReceived(0),
   m_dataStream( other.m_dataStream ),
   m_sSlot( other.m_sSlot ),
   m_cbMode( e_cbmode_stdout )
{
   m_class = other.m_class;
   m_context = 0;

   if ( other.m_handle != 0 )
      m_handle = curl_easy_duphandle( other.m_handle );
   else
      m_handle = 0;
}

CurlHandle::~CurlHandle()
{
   cleanup();
   if( m_context != 0 )
   {
      m_context->decref();
   }
}


bool CurlHandle::acquire( Process* prc )
{
   if(atomicCAS(m_inUse,0,1) == 1 ) {
      if( m_context != 0 )
      {
         if( m_context->process() != prc ) {
            m_context->decref();
            m_context = new WVMContext(prc);
         }
      }
      else {
         m_context = new WVMContext(prc);
      }

      return true;
   }

   return false;
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


      CurlList::iterator iter = m_slists.begin();
      while( iter != m_slists.end() )
      {
         struct curl_slist* slist = (struct curl_slist*) *iter;
         curl_slist_free_all( slist );
         ++iter;
      }
   }

   if (m_pPostBuffer != 0 )
   {
      free( m_pPostBuffer );
      m_pPostBuffer = 0;
   }
}


void CurlHandle::gcMark( uint32 mark )
{
   if( m_mark != mark )
   {
      m_iDataCallback.gcMark(mark);
      m_iReadCallback.gcMark(mark);

      if( m_sReceived != 0 )
         m_sReceived->gcMark( mark );

      m_mark = mark;
   }
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


size_t CurlHandle::write_string( void *ptr, size_t size, size_t nmemb, void *data)
{
   CurlHandle* h = (CurlHandle*) data;
   if ( h->m_sReceived == 0 )
      h->m_sReceived = new String( size * nmemb );

   String str;
   str.adopt( (char*) ptr, (int32) size * nmemb, 0 );
   h->m_sReceived->append( str );
   return size * nmemb;
}


size_t CurlHandle::write_callback( void *ptr, size_t size, size_t nmemb, void *data)
{
   CurlHandle* self = (CurlHandle*) data;

   String* str = new String;
   str->adopt( ( char*) ptr, (int32) size * nmemb, 0 );
   Item params[1];
   params[0] = FALCON_GC_HANDLE( str );

   // start the falcon routine in a parallel context.
   WVMContext* ctx = self->context();
   ctx->reset();
   ctx->startItem( self->m_iDataCallback, 1, params);
   // this might throw
   ctx->wait();

   const Item& result = ctx->result();

   if( result.isNil() || (result.isBoolean() && result.asBoolean() ) )
   {
      return size*nmemb;
   }
   else if( result.isOrdinal() )
   {
      return (size_t) result.forceInteger();
   }

   return 0;
}


void CurlHandle::setOutStream( Stream* stream )
{
   if( m_dataStream != 0 ){ m_dataStream->decref(); }
   if( stream != 0 ) {stream->incref();}
   m_dataStream = stream;
}


void CurlHandle::setInStream( Stream* stream )
{
   if( m_readStream != 0 ){ m_readStream->decref(); }
   if( stream != 0 ) {stream->incref();}
   m_readStream = stream;
}


void CurlHandle::setOnDataCallback( const Item& itm )
{
   m_sReceived = 0;
   setOutStream(0);

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

   setOutStream(s);
   s->incref();
   m_cbMode = e_cbmode_stream;

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_WRITEFUNCTION, write_stream );
      curl_easy_setopt( m_handle, CURLOPT_WRITEDATA, s );
   }
}



void CurlHandle::setOnDataGetString()
{
   // the string is initialized by the callback
   m_sReceived = 0;
   m_iDataCallback.setNil();
   setOutStream(0);

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
   setOutStream(0);

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
   setInStream(0);

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
   setInStream( stream );

   if( m_handle != 0 )
   {
      curl_easy_setopt( m_handle, CURLOPT_READFUNCTION, read_stream );
      curl_easy_setopt( m_handle, CURLOPT_READDATA, this );
   }
}


String* CurlHandle::getData()
{
   String* ret = m_sReceived;
   m_sReceived = 0;
   return ret;
}

size_t CurlHandle::read_callback( void *ptr, size_t size, size_t nmemb, void *data)
{
   CurlHandle* self = (CurlHandle*) data;
   WVMContext* ctx = self->context();

   String temp;
   temp.adoptMemBuf((byte*) ptr, size*nmemb, size*nmemb);
   Item params[2];
   params[0].setUser(temp.handler(), &temp);
   params[1].setInteger(((int64)size) * ((int64)nmemb));

   ctx->reset();
   ctx->startItem(self->m_iReadCallback, 2, params);
   ctx->wait();

   return (size_t) ctx->result().forceInteger();
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

struct curl_slist* CurlHandle::slistFromArray( ItemArray* ca )
{
   struct curl_slist* sl = NULL;

   for( uint32 pos = 0; pos < ca->length(); ++pos )
   {
      Item& current = ca->at(pos);
      if( ! current.isString() )
      {
         if( sl != 0 )
         {
            m_slists.push_back( sl );
         }
         return 0;
      }

      AutoCString str( *current.asString() );
      sl = curl_slist_append( sl, str.c_str() );
   }

   if( sl != 0 )
   {
      m_slists.push_back( sl );
   }

   return sl;
}


void CurlHandle::postData( const String& str )
{
   if (m_pPostBuffer != 0 )
      free( m_pPostBuffer );

   m_pPostBuffer = malloc( str.size() );
   memcpy(m_pPostBuffer, str.getRawStorage(), str.size() );

   curl_easy_setopt( handle(), CURLOPT_POSTFIELDS, m_pPostBuffer );
   curl_easy_setopt( handle(), CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t) str.size() );
}



//==============================================================
//

CurlMultiHandle::CurlMultiHandle()
{
   m_handle = curl_multi_init();
   m_mtx = new Mutex;
   m_refCount = new int(1);
}


CurlMultiHandle::CurlMultiHandle( const CurlMultiHandle &other )
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



void CurlMultiHandle::gcMark( uint32 mark )
{
   m_handles.gcMark( mark );
}


bool CurlMultiHandle::addHandle( CurlHandle* h )
{
   for ( uint32 i = 0; i < m_handles.length(); ++i )
   {
      if ( m_handles[i].asInst() == h )
         return false;
   }

   m_handles.append( Item( h->cls(), h) );
   curl_multi_add_handle( handle(), h->handle() );
   return true;
}


bool CurlMultiHandle::removeHandle( CurlHandle* h )
{
   for ( uint32 i = 0; i < m_handles.length(); ++i )
   {
      if ( m_handles[i].asInst() == h )
      {
         curl_multi_remove_handle( handle(), h->handle() );
         m_handles.remove( i );
         return true;
      }
   }

   return false;
}


//=============================================================================
//SimpleCurlRequest
//=============================================================================
SimpleCurlRequest::SimpleCurlRequest( CurlHandle* handle, ::Falcon::Process* prc ):
         m_curlHandle(handle)
{
   m_thread = new SysThread(this);
   m_complete = new Shared(&prc->vm()->contextManager());
   m_retval = CURLE_OK;
   m_error = 0;
}

SimpleCurlRequest::~SimpleCurlRequest()
{
   m_complete->decref();
}

void SimpleCurlRequest::start()
{
   m_thread->start(ThreadParams().detached(true));
}

void* SimpleCurlRequest::run()
{
   CURL* curl = m_curlHandle->handle();

   try {
      // callback functions can throw
      m_retval = curl_easy_perform(curl);
   }
   catch(::Falcon::Error* e )
   {
      m_error = e;
   }

   m_complete->signal();

   return 0;
}

}
}


/* end of curl_mod.cpp */
