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
   Internal logic functions - declarations.
*/

#ifndef curl_mod_H
#define curl_mod_H

#include <falcon/itemarray.h>
#include <falcon/error.h>
#include <falcon/shared.h>
#include <falcon/mt.h>
#include <falcon/wvmcontext.h>
#include <falcon/atomic.h>
#include <falcon/shared.h>
#include <curl/curl.h>

#include <list>

namespace Falcon {
namespace Mod {

class SimpleCurlRequest;

class CurlHandle
{
public:
   CurlHandle( const Class* maker);
   CurlHandle( const CurlHandle &other );

   virtual ~CurlHandle();
   virtual CurlHandle* clone() const;

   CURL* handle() const { return m_handle; }

   virtual void gcMark( uint32 mark );

   void setOnDataCallback( const Item& callable );
   void setOnDataStream( Stream* s );
   void setOnDataGetString();
   void setOnDataStdOut();

   void setReadCallback( const Item& callable );
   void setReadStream( Stream* read );

   String* getData();

   void cleanup();

   /** Creates a curl_slist from a Falcon array of strings.
    *  The list is stored here and destroyed by cleanup().
    *  Returns zero if some elements of ca are not strings.
    *
    */
   struct curl_slist* slistFromArray( ItemArray* ca );

   /** Stores data for post operations.
    * Saves a copy of the string in a local buffer, that is destroyed
    * at cleanup(), and sets the POSTFIELDS and CURLOPT_POSTFIELDSIZE_LARGE
    * options correctly.
    *
    * Multiple operations will cause the previous buffer to be discarded.
    */
   void postData( const String& str );

   uint32 currentMark() const { return m_mark; }

   /** Sets the request context that has caused this handle to be processed now.
    *
    * Used in callback functions to access the context information about
    * the request that caused this handle to be used.
    */
   void request( SimpleCurlRequest* r ) {m_ownerRequest = r; }
   SimpleCurlRequest* request() const {return m_ownerRequest; }

   /** Prevent execution from parallel processes.
    *  If true, the resource is acquired atomically; every other request
    *  before release() will fail.
    */
   bool acquire( Process* prc );
   void release() { atomicCAS(m_inUse,1,0); }

   // The handler class for this handle
   const Class* cls() const { return m_class; }

   WVMContext* context() const { return m_context; }

protected:
   /** Callback modes.
    *
    */
   typedef enum
   {
      e_cbmode_stdout,
      e_cbmode_string,
      e_cbmode_stream,
      e_cbmode_slot,
      e_cbmode_callback
   } t_cbmode;

private:

   static size_t write_stdout( void *ptr, size_t size, size_t nmemb, void *data);
   static size_t write_stream( void *ptr, size_t size, size_t nmemb, void *data);
   static size_t write_string( void *ptr, size_t size, size_t nmemb, void *data);
   static size_t write_callback( void *ptr, size_t size, size_t nmemb, void *data);

   static size_t read_callback( void *ptr, size_t size, size_t nmemb, void *data);
   static size_t read_stream( void *ptr, size_t size, size_t nmemb, void *data);

   CURL* m_handle;

   Item m_iDataCallback;
   String* m_sReceived;
   Stream* m_dataStream;
   String m_sSlot;
   const Class* m_class;

   /** Callback mode, determining which of the method to notify the app is used. */
   t_cbmode m_cbMode;

   Item m_iReadCallback;
   Stream* m_readStream;

   // lists of lists to be destroyed at exit.
   typedef std::list<curl_slist*> CurlList;
   CurlList m_slists;

   void* m_pPostBuffer;

   uint32 m_mark;
   SimpleCurlRequest* m_ownerRequest;

   atomic_int m_inUse;


   WVMContext* m_context;

   void setOutStream( Stream* stream );
   void setInStream( Stream* stream );
};



class CurlMultiHandle
{
public:
   CurlMultiHandle();
   CurlMultiHandle( const CurlMultiHandle &other );

   virtual ~CurlMultiHandle();
   virtual CurlMultiHandle* clone() const;

   CURLM* handle() const { return m_handle; }

   virtual void gcMark( uint32 mark );

   bool addHandle( CurlHandle* h );
   bool removeHandle( CurlHandle* );

   uint32 currentMark() const { return m_mark; }

   bool acquireAll(VMContext* ctx);
   void releaseAll();

private:
   CURLM* m_handle;
   Mutex* m_mtx;
   int* m_refCount;

   ItemArray m_handles;
   uint32 m_mark;
};


class SimpleCurlRequest: public Runnable
{
public:
   SimpleCurlRequest( CurlHandle* handle, ::Falcon::Process* prc );
   virtual ~SimpleCurlRequest();

   virtual void* run();

   void start();

   Error* exitError() const { return m_error; }
   CURLcode exitCode() const { return m_retval; }

   Shared& complete() const {return *m_complete; }
private:

   mutable Shared* m_complete;
   SysThread* m_thread;
   CurlHandle* m_curlHandle;

   CURLcode m_retval;
   Error* m_error;
};

}
}

#endif

/* end of curl_mod.h */
