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
   Interface extension functions
*/

#ifndef _MSC_VER
#include <sys/time.h>
#else 
#include <time.h>
   #ifndef _WIN32_WINNT
      #define _WIN32_WINNT 0x0403 
   #elif _WIN32_WINNT < 0x0403
      #undef _WIN32_WINNT
      #define _WIN32_WINNT 0x0403 
   #endif
#endif

#include <falcon/stdhandlers.h>
#include <falcon/autocstring.h>
#include <falcon/uri.h>
#include <falcon/itemarray.h>
#include <falcon/itemdict.h>
#include <falcon/timestamp.h>

#include <curl/curl.h>

#include "curl_mod.h"
#include "curl_ext.h"
#include "curl_fm.h"
#include "curl_st.h"

#undef SRC
#define SRC "modules/native/curl/curl_ext.cpp"

/*# @beginmodule curl
*/

using namespace Falcon::Canonical;

namespace Falcon {
namespace Curl {

static void throw_error( int code, int line, const String& cd, const CURLcode retval )
{
   String error = String( curl_easy_strerror( retval ) );
   throw new CurlError( ErrorParam( code, line, SRC )
         .desc( cd )
         .extra( error.A(" (").N(retval).A(")") )
         );
}

static void throw_merror( int code, int line, const String& cd, const CURLMcode retval )
{
   String error = String( curl_multi_strerror( retval ) );
   throw new CurlError( ErrorParam( code, line, SRC )
         .desc( cd )
         .extra( error.A(" (").N(retval).A(")") )
         );
}

static void internal_curl_init( Function* func, VMContext* , Mod::CurlHandle* h, Item* i_uri )
{
   Class* clsURI = Engine::instance()->stdHandlers()->uriClass();

   CURL* curl = h->handle();

   // we had a general init error from curl
   if ( curl == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_INIT, __LINE__, SRC )
                  .desc( curl_err_init )
                  .extra( curl_err_resources  ) );
   }

   curl_easy_setopt( curl, CURLOPT_NOPROGRESS, 1 );
   curl_easy_setopt( curl, CURLOPT_NOSIGNAL, 1 );

   // we need ourselves inside the object, so we can handle us back in multiple.
   curl_easy_setopt( curl, CURLOPT_PRIVATE, h );


   // no parameter? -- nothing to do
   if( i_uri == 0 )
      return;

   CURLcode retval;

   if( i_uri->isString() )
   {
      //String enc = URI::URLEncode( *i_uri->asString() );
      AutoCString curi( *i_uri->asString() );

      retval = curl_easy_setopt( curl, CURLOPT_URL, curi.c_str() );
   }
   else if( i_uri->isInstanceOf( clsURI ) )
   {
      URI* uri = (URI*) i_uri->asInst();
      AutoCString curi( uri->encode() );

      retval = curl_easy_setopt( curl, CURLOPT_URL, curi.c_str() );
   }
   else
   {
      throw func->paramError();
   }

   if( retval != CURLE_OK )
   {
      throw_error( FALCON_ERROR_CURL_INIT, __LINE__, curl_err_init, retval );
   }
}

/*#
   @class Handle
   @brief Stores an handle for a CURL (easy) connection.
   @optparam uri A string or an URI to be used to initialize the connection.

   The following is a simple complete program that retrieves the main
   page of the Falcon programming language site:

   @code
   import from curl

   try
      h = curl.Handle( "http://www.falconpl.org" )
      h.setOutString()
      h.exec()
      > "Complete data transfer:", h.getData()
   catch curl.CurlError in e
      > "ERROR: ", e
   end
   @endcode

   @prop data User available data. Store any data that the client application
         wants to be related to this handle in this property. It is also possible
         to derive a child from this
         class and store more complete behavior there.
*/
namespace CHandle {

FALCON_DECLARE_FUNCTION(init, "uri:[S|Uri]")
FALCON_DEFINE_FUNCTION_P1(init)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();
   Item* i_uri = ctx->param(0);

   internal_curl_init( this, ctx, h, i_uri );
   ctx->returnFrame(ctx->self());
}

/*#
   @method exec Handle
   @brief Transfers data from the remote.
   @return self (to put this call in a chain)
   @raise CurlError on error


   This function performs the whole transfer towards the target that has been
   selected via @a Handle.setOutString, @a Handle.setOutStream, @a Handle.setOutConsole
   or @a Handle.setOutCallback routines.

   The call is blocking and normally it cannot be interrupted; however, a
   timeout can be set through

   @note Internally, this method performs a curl_easy_perform call on the inner

*/
FALCON_DECLARE_FUNCTION(exec, "")
FALCON_DEFINE_FUNCTION_P1(exec)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();
   CURL* curl = h->handle();

   if ( curl == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   if (! h->acquire(ctx->process()) )
   {
      throw FALCON_SIGN_XERROR(CodeError, e_concurrence, .extra("Curl handle"));
   }

   // prepare to return when we're done.
   const ClassHandle* cls = static_cast<const ClassHandle*>(h->cls());
   ctx->pushCode( cls->afterExec() );

   // prepare a request that will go in parallel...
   Mod::SimpleCurlRequest* sr = new Mod::SimpleCurlRequest(h, ctx->process());
   h->request(sr);
   sr->start();

   // and tell the engine we'll wait till complete.
   ctx->addWait( &sr->complete() );
   ctx->engageWait(-1);
}

// The PSTEP used to return the value of the exec operation.
class PStepAfterExec: public PStep
{
public:
   PStepAfterExec() {apply = apply_;}
   virtual ~PStepAfterExec() {}

   virtual void describeTo(String& target) const {
      target = "Curl::ClassHandle::PStepAfterExec";
   }

   static void apply_(const PStep*, VMContext* ctx )
   {
      // get the request and "release" the old request.
      Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();
      Mod::SimpleCurlRequest* r = h->request();
      h->request(0);

      // get the infos about the request and delete it.
      Error* error = r->exitError();
      CURLcode retval = r->exitCode();
      delete r;

      // free the handle for new requests
      h->release();

      if( r->exitError() != 0 )
      {
         throw error;
      }

      if( retval != CURLE_OK )
      {
         throw_error( FALCON_ERROR_CURL_EXEC, __LINE__, curl_err_exec, retval );
      }

      ctx->returnFrame( ctx->self() );
   }
};


/*#
   @method setOutConsole Handle
   @brief Asks for subsequent transfer(s) to be sent to process console (raw stdout).
   @return self (to put this call in a chain)

   This is the default at object creation.
*/

FALCON_DECLARE_FUNCTION(setOutConsole, "")
FALCON_DEFINE_FUNCTION_P1(setOutConsole)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   h->setOnDataStdOut();
}

/*#
   @method setOutString Handle
   @brief Asks for subsequent transfer(s) to be stored in a temporary string.
   @return self (to put this call in a chain)


   After @a Handle.exec has been called, the data will be available in
   a string that can be retrieved via the @a Handle.getData method.

*/
FALCON_DECLARE_FUNCTION(setOutString, "")
FALCON_DEFINE_FUNCTION_P1(setOutString)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );

   h->setOnDataGetString();
   ctx->returnFrame( ctx->self() );
}


/*#
   @method setOutStream Handle
   @brief Asks for subsequent transfer(s) to be stored in a given stream.
   @param stream The stream to be used.
   @return self (to put this call in a chain)

   When called, @a Handle.exec will store incoming data in this stream object
   via binary Stream.write operations.
*/

FALCON_DECLARE_FUNCTION(setOutStream, "stream:Stream")
FALCON_DEFINE_FUNCTION_P1(setOutStream)
{
   static Class* clsStream = Engine::instance()->stdHandlers()->streamClass();

   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   Item* i_stream = ctx->param(0);

   if ( i_stream == 0 || ! i_stream->isInstanceOf(clsStream) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "Stream" ) );
   }

   h->setOnDataStream( (Stream*) i_stream->asInst() );
   ctx->returnFrame( ctx->self() );
}

/*#
   @method setOutCallback Handle
   @brief Asks for subsequent transfer(s) to be handled to a given callback.
   @param cb A callback item that will receive incoming data as a binary string.
   @return self (to put this call in a chain)

   This method instructs this handle to call a given callback when data
   is received.

   When called, @a Handle.exec will repeatedly call the @b cb item providing
   a single string as a parameter.

   The string is not encoded in any format, and could be considered filled with binary
   data.
*/
FALCON_DECLARE_FUNCTION(setOutCallback, "cb:C")
FALCON_DEFINE_FUNCTION_P1(setOutCallback)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   Item* i_cb = ctx->param(0);

   if ( i_cb  == 0 || ! i_cb->isCallable() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "C" ) );
   }

   h->setOnDataCallback( *i_cb );
   ctx->returnFrame( ctx->self() );
}

/*#
   @method cleanup Handle
   @brief Close a connection and destroys all associated data.

   After this call, the handle is not usable anymore.
   This is executed also automatically at garbage collection, but
   the user may be interested in clearing the data as soon as possible.
*/
FALCON_DECLARE_FUNCTION(cleanup, "")
FALCON_DEFINE_FUNCTION_P1(cleanup)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   h->cleanup();
   ctx->returnFrame( );
}

/*#
   @method setInCallback Handle
   @brief Asks for subsequent uploads to be handled to a given callback.
   @param cb A callback item that will write data in an incoming MemBuf
   @return self (to put this call in a chain)

   This method instructs this handle to call a given callback when new
   data can be uploaded to the remote side.

   The function receives a MemBuf that must be filled with data.

   It should return the amount of data really written to the membuf.

   It can also return CURL.WRITE_PAUSE to ask for @a Handle.exec to return with
   a pause status.

   The callback must return 0 when it has no more data to transfer.
*/

FALCON_DECLARE_FUNCTION(setInCallback, "cb:C")
FALCON_DEFINE_FUNCTION_P1(setInCallback)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   Item* i_cb = ctx->param(0);

   if ( i_cb  == 0 || ! i_cb->isCallable() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "C" ) );
   }

   h->setReadCallback( *i_cb );
   ctx->returnFrame( ctx->self() );
}


/*#
   @method setInStream Handle
   @brief Asks for subsequent upload(s) to read data from the given stream.
   @param stream The stream to be used.
   @return self (to put this call in a chain)

   When called, @a Handle.exec will read data to be uploaded from this
   stream.
*/
FALCON_DECLARE_FUNCTION(setInStream, "stream:Stream")
FALCON_DEFINE_FUNCTION_P1(setInStream)
{
   static Class* clsStream = Engine::instance()->stdHandlers()->streamClass();

   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   Item* i_stream = ctx->param(0);

   if ( i_stream == 0 || ! i_stream->isInstanceOf(clsStream) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "Stream" ) );
   }

   h->setReadStream( (Stream*) i_stream->asInst() );
   ctx->returnFrame( ctx->self() );
}

/*#
   @method getData Handle
   @brief Gets data temporarily stored in a string during a transfer.
   @return A string containing data that has been transfered.

   This function returns the data received in the meanwhile. This data
   is captured when the @a Handle.setOutString option has been set.

*/
FALCON_DECLARE_FUNCTION(getData, "")
FALCON_DEFINE_FUNCTION_P1(getData)
{
   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   String* s = h->getData();
   if( s != 0 )
   {
      ctx->returnFrame( FALCON_GC_HANDLE( s ) );
   }
   else {
      ctx->returnFrame();
   }
}


static void internal_setOpt( VMContext*, Mod::CurlHandle* h, CURLoption iOpt, Item* i_data )
{
   CURLcode ret;
   CURL* curl = h->handle();

   switch( iOpt )
   {
   case CURLOPT_VERBOSE:
   case CURLOPT_HEADER:
   case CURLOPT_NOPROGRESS:
   case CURLOPT_HTTPPROXYTUNNEL:
#if LIBCURL_VERSION_NUM >= 0x071904
   case CURLOPT_SOCKS5_GSSAPI_NEC:
#endif
   case CURLOPT_TCP_NODELAY:
   case CURLOPT_AUTOREFERER:
   case CURLOPT_FOLLOWLOCATION:
   case CURLOPT_UNRESTRICTED_AUTH:
   case CURLOPT_PUT:
   case CURLOPT_POST:
   case CURLOPT_COOKIESESSION:
   case CURLOPT_HTTPGET:
   case CURLOPT_IGNORE_CONTENT_LENGTH:
   case CURLOPT_DIRLISTONLY:
   case CURLOPT_APPEND:
   case CURLOPT_FTP_USE_EPRT:
   case CURLOPT_FTP_USE_EPSV:
   case CURLOPT_FTP_CREATE_MISSING_DIRS:
   case CURLOPT_FTP_SKIP_PASV_IP:
   case CURLOPT_TRANSFERTEXT:
   case CURLOPT_PROXY_TRANSFER_MODE:
   case CURLOPT_CRLF:
   case CURLOPT_FILETIME:
   case CURLOPT_NOBODY:
   case CURLOPT_UPLOAD:
   case CURLOPT_FRESH_CONNECT:
   case CURLOPT_FORBID_REUSE:
   case CURLOPT_CONNECT_ONLY:
   case CURLOPT_SSLENGINE_DEFAULT:
   case CURLOPT_SSL_VERIFYPEER:
#if LIBCURL_VERSION_NUM >= 0x071901
   case CURLOPT_CERTINFO:
#endif
   case CURLOPT_SSL_SESSIONID_CACHE:
      {
         long bVal = i_data->isTrue() ? 1 : 0;
         ret = curl_easy_setopt( curl, iOpt, bVal );
      }
      break;
   case CURLOPT_SSL_VERIFYHOST:
     {
       long bVal = i_data->isTrue() ? 2 : 0;
       ret = curl_easy_setopt( curl, iOpt, bVal );
     }
       break;
#ifdef CURLOPT_PROTOCOLS
    case CURLOPT_PROTOCOLS:
    case CURLOPT_REDIR_PROTOCOLS:
#endif
    case CURLOPT_PROXYPORT:
    case CURLOPT_PROXYTYPE:
    case CURLOPT_LOCALPORT:
    case CURLOPT_LOCALPORTRANGE:
    case CURLOPT_DNS_CACHE_TIMEOUT:
    case CURLOPT_DNS_USE_GLOBAL_CACHE:
    case CURLOPT_BUFFERSIZE:
    case CURLOPT_PORT:
#if LIBCURL_VERSION_NUM >= 0x071900
    case CURLOPT_ADDRESS_SCOPE:
#endif
    case CURLOPT_NETRC:
    case CURLOPT_HTTPAUTH:
    case CURLOPT_PROXYAUTH:
    case CURLOPT_MAXREDIRS:
#if LIBCURL_VERSION_NUM >= 0x071901
    case CURLOPT_POSTREDIR:
#endif
    case CURLOPT_HTTP_VERSION:
    case CURLOPT_HTTP_CONTENT_DECODING:
    case CURLOPT_HTTP_TRANSFER_DECODING:
#if LIBCURL_VERSION_NUM >= 0x071904
    case CURLOPT_TFTP_BLKSIZE:
#endif
    case CURLOPT_FTP_RESPONSE_TIMEOUT:
    case CURLOPT_USE_SSL:
    case CURLOPT_FTPSSLAUTH:
    case CURLOPT_FTP_SSL_CCC:
    case CURLOPT_FTP_FILEMETHOD:
    case CURLOPT_RESUME_FROM:
    case CURLOPT_INFILESIZE:
    case CURLOPT_MAXFILESIZE:
    case CURLOPT_TIMECONDITION:
    case CURLOPT_TIMEVALUE:
    case CURLOPT_TIMEOUT:
    case CURLOPT_TIMEOUT_MS:
    case CURLOPT_LOW_SPEED_LIMIT:
    case CURLOPT_LOW_SPEED_TIME:
    case CURLOPT_MAXCONNECTS:
    case CURLOPT_CONNECTTIMEOUT:
    case CURLOPT_CONNECTTIMEOUT_MS:
    case CURLOPT_IPRESOLVE:
    case CURLOPT_SSLVERSION:
    case CURLOPT_SSH_AUTH_TYPES:
    case CURLOPT_NEW_FILE_PERMS:
    case CURLOPT_NEW_DIRECTORY_PERMS:
       {
          long lVal = (long) i_data->asInteger();
          ret = curl_easy_setopt( curl, iOpt, lVal );
       }
       break;

    case CURLOPT_RESUME_FROM_LARGE:
    case CURLOPT_INFILESIZE_LARGE:
    case CURLOPT_MAXFILESIZE_LARGE:
    case CURLOPT_MAX_SEND_SPEED_LARGE:
    case CURLOPT_MAX_RECV_SPEED_LARGE:
       {
          if( ! i_data->isOrdinal() )
          {
             throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                   .extra( curl_err_setopt ));
          }

          curl_off_t offset = (curl_off_t) i_data->forceInteger();
          ret = curl_easy_setopt( curl, iOpt, offset );
       }
       break;



    case CURLOPT_URL:
    case CURLOPT_PROXY:
#if LIBCURL_VERSION_NUM >= 0x071904
    case CURLOPT_NOPROXY:
    case CURLOPT_SOCKS5_GSSAPI_SERVICE:
#endif
    case CURLOPT_INTERFACE:
    case CURLOPT_NETRC_FILE:
    case CURLOPT_USERPWD:
    case CURLOPT_PROXYUSERPWD:
#if LIBCURL_VERSION_NUM >= 0x071901
    case CURLOPT_USERNAME:
    case CURLOPT_PASSWORD:
    case CURLOPT_PROXYUSERNAME:
    case CURLOPT_PROXYPASSWORD:
#endif
    case CURLOPT_ENCODING:
    case CURLOPT_REFERER:
    case CURLOPT_USERAGENT:
    case CURLOPT_COOKIE:
    case CURLOPT_COOKIEFILE:
    case CURLOPT_COOKIEJAR:
    case CURLOPT_COOKIELIST:
    case CURLOPT_FTPPORT:
    case CURLOPT_FTP_ALTERNATIVE_TO_USER:
    case CURLOPT_FTP_ACCOUNT:
    case CURLOPT_RANGE:
    case CURLOPT_CUSTOMREQUEST:
    case CURLOPT_SSLCERT:
    case CURLOPT_SSLCERTTYPE:
    case CURLOPT_SSLKEY:
    case CURLOPT_SSLKEYTYPE:
    case CURLOPT_KEYPASSWD:
    case CURLOPT_SSLENGINE:
    case CURLOPT_CAINFO:
#if LIBCURL_VERSION_NUM >= 0x071900
    case CURLOPT_ISSUERCERT:
    case CURLOPT_CRLFILE:
#endif
    case CURLOPT_CAPATH:
    case CURLOPT_RANDOM_FILE:
    case CURLOPT_EGDSOCKET:
    case CURLOPT_SSL_CIPHER_LIST:
    case CURLOPT_KRBLEVEL:
    case CURLOPT_SSH_HOST_PUBLIC_KEY_MD5:
    case CURLOPT_SSH_PUBLIC_KEYFILE:
    case CURLOPT_SSH_PRIVATE_KEYFILE:
#ifdef CURLOPT_SSH_KNOWNHOSTS
    case CURLOPT_SSH_KNOWNHOSTS:
#endif
       {
           if( ! i_data->isString() )
           {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                    .extra( curl_err_setopt ) );
           }

           AutoCString cstr( *i_data->asString() );
           ret = curl_easy_setopt( curl, iOpt, cstr.c_str() );
        }
        break;

    case CURLOPT_HTTPHEADER:
    case CURLOPT_HTTP200ALIASES:
    case CURLOPT_QUOTE:
    case CURLOPT_POSTQUOTE:
    case CURLOPT_PREQUOTE:
       {
          if( ! i_data->isArray() )
          {
              throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                    .extra( curl_err_setopt ) );
          }

          ItemArray* items = i_data->asArray();
          struct curl_slist *slist = h->slistFromArray( items );
          if ( slist == 0 )
          {
               throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                     .extra( curl_err_setopt ) );
          }

          ret = curl_easy_setopt( curl, iOpt, slist );
       }
       break;

    default:
       throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
             .extra( curl_err_unkopt ) );
   }

   if( ret != CURLE_OK )
   {
      throw_error( FALCON_ERROR_CURL_SETOPT, __LINE__, curl_err_setopt, ret );
   }

}


/*#
   @method setOption Handle
   @brief Sets a cURL option for this specific handle.
   @param option The option to be set (an enumeration).
   @param data The value to be set.
   @return self (to put this call in a chain)

   Depending on the option, @b data must be a boolean, a number or
   a string.

   Some options, as CURLOPT.HTTPHEADER, require the data to be an array
   of strings.

   @note CURLOPT.POSTFIELDS family options are not supported directly by
   this function; use @a Handle.postData function instead.

   @note Callback related options are not supported by this function.
   Specific functions are provided to setup automated or manual callback
   facilities (see the various set* methods in this class).
 */

FALCON_DECLARE_FUNCTION(setOption, "option:N,data:X")
FALCON_DEFINE_FUNCTION_P1(setOption)
{
   Item* i_option = ctx->param(0);
   Item* i_data = ctx->param(1);

   if ( i_option == 0 || ! i_option->isInteger()
         || i_data == 0 )
   {
      throw paramError();
   }

   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   CURLoption iOpt = (CURLoption) i_option->asInteger();
   internal_setOpt( ctx, h, iOpt, i_data );
   ctx->returnFrame( ctx->self() );
}


/*#
   @method setOptions Handle
   @brief Sets a list of cURL option for this specific handle.
   @param opts A dictionary of options, where each key is an option number, and its value is the option value.
   @return self (to put this call in a chain)
*/
FALCON_DECLARE_FUNCTION(setOptions, "otps:D")
FALCON_DEFINE_FUNCTION_P1(setOptions)
{
   Item* i_opts = ctx->param(0);

   if ( i_opts == 0 || ! i_opts->isDict() )
   {
      throw paramError();
   }

   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__, SRC )
            .desc( curl_err_pm ) );
   }

   ItemDict* dict = i_opts->asDict();

   class Rator: public ItemDict::Enumerator
   {
   public:
      Rator( VMContext* c, Mod::CurlHandle* h ): ctx(c), m_h(h) {}
      virtual ~Rator(){}

      virtual void operator()( const Item& key, Item& value )
      {
         if( ! key.isInteger() )
         {
            throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
                        .extra( "D[I=>X]" ) );
         }

         internal_setOpt( ctx, m_h, (CURLoption) key.asInteger(), &value );
      }
   private:
      VMContext* ctx;
      Mod::CurlHandle* m_h;
   }
   rator(ctx,h);

   dict->enumerate(rator);
   ctx->returnFrame(ctx->self());
}


/*#
   @method postData Handle
   @brief Sets data to be sent in one unique POST request.
   @param data A string to be sent as post data.

   This function substitutes the CURLOPT_POSTFIELDS family of options
   of the C level libcurl. It allows to set a string that will be
   sent in HTTP post request.

   All the other setOut* methods can be used for the same purpose to
   take data from streams, callback or even strings, but all the other
   methods will transfer data in chunks, and require to set the HTTP header
   transfer-encoding as "chunked" via the CURL.HTTP_HEADERS option, and
   to use HTTP/1.1 protocol.

   Using this method, the postData will be sent as an unique chunk, so
   it doesn't require extra header setting and works with any HTTP protocol.

   @note The data will be sent not encoded in any particular format (it will be
         binary-transmitted as it is in the string memory). If the remote
         server expects a particular encoding (usually, UTF-8), appropriate
         transocoding functions must be used in advance.
*/

FALCON_DECLARE_FUNCTION(postData, "data:S")
FALCON_DEFINE_FUNCTION_P1(postData)
{
   Item* i_data = ctx->param(0);

   if ( i_data == 0 || ! i_data->isString() )
   {
     throw paramError();
   }

   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
     throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
           .desc( curl_err_pm ) );
   }

   h->postData( *i_data->asString() );
   ctx->returnFrame(ctx->self());
}

/*#
   @method getInfo Handle
   @brief Returns informations about the status of this handle.
   @param option The specific information to be read.
   @return The value associated with the required information, or
           nil if the option is not available.

   This method returns one of the informations that can be
   retrieved from this handle. The option value are stored in the
   INFO enumeration, and they correspond to the values in the
   CURLINFO_* set of defines of the libcurl SDK, associated with the
   curl_easy_getinfo function.

   The type of the returned value depends of the type of information
   required; in general it may be a number or a string.

   Possible values for @b option are

   - INFO.EFFECTIVE_URL - the last used effective URL.
   - INFO.RESPONSE_CODE - the last received HTTP or FTP code. This will be zero if no server response
      code has been received. Note that a proxy's CONNECT response should be read with
      INFO.HTTP_CONNECTCODE and not this.
   - INFO.HTTP_CONNECTCODE - the last received proxy response code to a CONNECT request.
   - INFO.FILETIME - time of the retrieved document (as a Falcon TimeStamp, in GMT). If you get @b nil, it can be because of many reasons
     (unknown, the server hides it or the server doesn't support the command that tells document time etc)
     and the time of the document is unknown. Note that you must tell the server to collect this information before the transfer
     is made, by using the OPT.FILETIME option to @a Handle.setOption or you will unconditionally get a @b nil back.
   - INFO.TOTAL_TIME - the total time in seconds and fractions for the previous transfer, including name resolving, TCP connect etc.
   - INFO.NAMELOOKUP_TIME - time, in seconds and fractions, it took from the start until the name resolving was completed.
   - INFO.CONNECT_TIME - time, in seconds and fraction, it took from the start until the connect to the remote host (or proxy) was completed.
   - INFO.APPCONNECT_TIME - time, in seconds and fractions, it took from the start until the SSL/SSH connect/handshake to the remote host was completed. This time is most often very near to the PRETRANSFER time, except for cases such as HTTP pippelining where the pretransfer time can be delayed due to waits in line for the pipeline and more. (Added in 7.19.0)
   - INFO.PRETRANSFER_TIME - time, in seconds and fractions, it took from the start until the file transfer is just about to begin. This includes all pre-transfer commands and negotiations that are specific to the particular protocol(s) involved.
   - INFO.STARTTRANSFER_TIME - time, in seconds and fractions, it took from the start until the first byte is just about to be transferred. This includes CURLINFO_PRETRANSFER_TIME and also the time the server needs to calculate the result.
   - INFO.REDIRECT_TIME - total time, in seconds and fractions, it took for all redirection steps include name lookup, connect, pretransfer and transfer before final transaction was started. CURLINFO_REDIRECT_TIME contains the complete execution time for multiple redirections. (Added in 7.9.7)
   - INFO.REDIRECT_COUNT - total number of redirections that were actually followed.
   - INFO.REDIRECT_URL - the URL a redirect would take you to if you would enable OPT.FOLLOWLOCATION. This can come very handy if you
        think using the built-in libcurl redirect logic isn't good enough for you but you would still prefer to avoid implementing
        all the magic of figuring out the new URL.
   - INFO.SIZE_UPLOAD - total amount of bytes that were uploaded.
   - INFO.SIZE_DOWNLOAD - total amount of bytes that were downloaded. The amount is only for the latest transfer and will be reset again for each new transfer.
   - INFO.SPEED_DOWNLOAD - average download speed that curl measured for the complete download. Measured in bytes/second.
   - INFO.SPEED_UPLOAD - average upload speed that curl measured for the complete upload. Measured in bytes/second.
   - INFO.HEADER_SIZE - total size of all the headers received. Measured in number of bytes.
   - INFO.REQUEST_SIZE - total size of the issued requests. This is so far only for HTTP requests.
        Note that this may be more than one request if OPT.FOLLOWLOCATION is true.
   - INFO.SSL_VERIFYRESULT - the certification verification that was requested
       (using the OPT.SSL_VERIFYPEER option to @a Handle.setOption).
   - INFO.SSL_ENGINES - Array of OpenSSL crypto-engines supported. Note that engines are normally implemented in
         separate dynamic libraries. Hence not all the returned engines may be available at run-time.
   - INFO.CONTENT_LENGTH_DOWNLOAD - content-length of the download. This is the value read from the Content-Length:
         field. Since 7.19.4, this returns -1 if the size isn't known.
   - INFO.CONTENT_LENGTH_UPLOAD - specified size of the upload. Since 7.19.4, this returns -1 if the size isn't known.
   - INFO.CONTENT_TYPE - Pass a pointer to a char pointer to receive the content-type of the downloaded object.
      This is the value read from the Content-Type: field. If you get @b nil, it means that the server didn't send a valid Content-Type
      header or that the protocol used doesn't support this.

   - INFO.HTTPAUTH_AVAIL - bitmask indicating the authentication method(s) available. The meaning of the bits is explained in the
      OPT.HTTPAUTH option for @a Handle.setOption.

   - INFO.PROXYAUTH_AVAIL - bitmask indicating the authentication method(s) available for your proxy authentication.
   - INFO.OS_ERRNO - @b errno variable from a connect failure. Note that the value is only set on failure, it is not reset upon a
         successful operation.
   - INFO.NUM_CONNECTS - count of connections libcurl had to create to achieve the previous transfer
      (only the successful connects are counted).
      Combined with INFO.REDIRECT_COUNT you are able to know how many times libcurl successfully reused existing connection(s) or not.
   - INFO.PRIMARY_IP - IP address of the most recent connection done with this curl handle. This string may be IPv6 if that's enabled.
   - INFO_COOKIELIST - Returns an array of all the known cookies
   - INFO_FTP_ENTRY_PATH - FTP entry path. That is the initial path libcurl ended up in when logging on to the remote FTP server.
      This stores a @b nil if something is wrong.
   - INFO.CONDITION_UNMET - 1 if the condition provided in the previous request didn't match (see OPT.TIMECONDITION).
      Alas, if this returns a 1 you know that the reason you didn't get data in return is because it didn't
      fulfill the condition. The long this argument points to will get a zero stored if the condition instead was met.

*/
FALCON_DECLARE_FUNCTION(getInfo, "option:N")
FALCON_DEFINE_FUNCTION_P1(getInfo)
{
   Item* i_option = ctx->param(0);
   if( i_option == 0 || ! i_option->isOrdinal() )
   {
      throw paramError();
   }

   // setup our options
   Mod::CurlHandle* h = ctx->tself<Mod::CurlHandle*>();

   if ( h->handle() == 0 )
   {
     throw new CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
           .desc( curl_err_pm ) );
   }

   CURLINFO info = (CURLINFO) i_option->forceInteger();
   CURLcode cerr = CURLE_OK;

   switch( info )
   {

   // char*
   case CURLINFO_EFFECTIVE_URL:
   case CURLINFO_CONTENT_TYPE:
#if LIBCURL_VERSION_NUM >= 0x071900
   case CURLINFO_PRIMARY_IP:
#endif
   case CURLINFO_FTP_ENTRY_PATH:
   {
      char* rv;
      cerr = curl_easy_getinfo( h->handle(), info, &rv );
      if( cerr == CURLE_OK && rv != 0 )
      {
         String* cs = new String();
         cs->bufferize( rv );
         ctx->returnFrame( FALCON_GC_HANDLE(cs) );
      }
      // else we throw an error, so no return frame needed.
   }
   break;

   // long
   case CURLINFO_RESPONSE_CODE:
   case CURLINFO_HTTP_CONNECTCODE:
   case CURLINFO_HEADER_SIZE:
   case CURLINFO_REQUEST_SIZE:
   case CURLINFO_SSL_VERIFYRESULT:
   case CURLINFO_HTTPAUTH_AVAIL:
   case CURLINFO_PROXYAUTH_AVAIL:
   case CURLINFO_OS_ERRNO:
   case CURLINFO_NUM_CONNECTS:
#if LIBCURL_VERSION_NUM >= 0x071904
   case CURLINFO_CONDITION_UNMET:
#endif
      {
         long rv;
         cerr = curl_easy_getinfo( h->handle(), info, &rv );
         if( cerr == CURLE_OK )
         {
            ctx->returnFrame( (int64) rv );
         }
         // else we throw an error, so no return frame needed.
      }
      break;

      // timestamp
   case CURLINFO_FILETIME:
      {
         static Class* clsTS = Engine::instance()->stdHandlers()->timestampClass();

         long rv;
         cerr = curl_easy_getinfo( h->handle(), info, &rv );
         if( cerr == CURLE_OK && rv != -1 )
         {
            time_t trv = (time_t)rv;
            TimeStamp* timestamp = new TimeStamp;
            timestamp->timeZone(TimeStamp::tz_UTC );

            #ifndef FALCON_SYSTEM_WIN
               struct tm rtm;
               struct tm *ftime = gmtime_r( &trv, &rtm );

            #else
               struct tm *ftime = gmtime( &trv );
            #endif

            timestamp->year( ftime->tm_year + 1900 );
            timestamp->month( ftime->tm_mon + 1 );
            timestamp->day( ftime->tm_mday);
            timestamp->hour( ftime->tm_hour);
            timestamp->minute( ftime->tm_min);
            timestamp->second( ftime->tm_sec);
            timestamp->msec( 0);

            ctx->returnFrame(FALCON_GC_STORE(clsTS, timestamp));
         }
         // else we throw an error, so no return frame needed.
      }
      break;


      // double
   case CURLINFO_TOTAL_TIME:
   case CURLINFO_NAMELOOKUP_TIME:
   case CURLINFO_CONNECT_TIME:
#if LIBCURL_VERSION_NUM >= 0x071900
   case CURLINFO_APPCONNECT_TIME:
#endif
   case CURLINFO_PRETRANSFER_TIME:
   case CURLINFO_STARTTRANSFER_TIME:
   case CURLINFO_REDIRECT_TIME:
   case CURLINFO_REDIRECT_COUNT:
   case CURLINFO_REDIRECT_URL:
   case CURLINFO_SIZE_UPLOAD:
   case CURLINFO_SIZE_DOWNLOAD:
   case CURLINFO_SPEED_DOWNLOAD:
   case CURLINFO_SPEED_UPLOAD:
   case CURLINFO_CONTENT_LENGTH_DOWNLOAD:
   case CURLINFO_CONTENT_LENGTH_UPLOAD:
      {
         double rv;
         cerr = curl_easy_getinfo( h->handle(), info, &rv );
         if( cerr == CURLE_OK )
         {
           ctx->returnFrame( (numeric) rv );
         }
         // else we throw an error, so no return frame needed.
      }
      break;

      // slist
   case CURLINFO_SSL_ENGINES:
   case CURLINFO_COOKIELIST:
      {
         curl_slist* rv;
         cerr = curl_easy_getinfo( h->handle(), info, &rv );
         if( cerr == CURLE_OK )
         {
            ItemArray* ca = new ItemArray;
            curl_slist* p = rv;
            while( p != 0 )
            {
               ca->append( FALCON_GC_HANDLE( new String( p->data, String::npos ) ) );
               p = p->next;
            }

            curl_slist_free_all( rv );
            ctx->returnFrame( FALCON_GC_HANDLE(ca) );
         }
      }
      break;

   default:
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ) );
   }

   if( cerr != CURLE_OK )
   {
      throw_error( FALCON_ERROR_CURL_GETINFO, __LINE__, curl_err_getinfo, cerr );
   }
}

}


ClassHandle::ClassHandle():
         Class("Handle")
{
   setConstuctor(new CHandle::FALCON_FUNCTION_NAME(init) );

   addMethod(new CHandle::FALCON_FUNCTION_NAME(exec));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setOutConsole));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setOutString));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setOutStream));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setOutCallback));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(cleanup));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setInCallback));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setInStream));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(getData));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setOption));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(setOptions));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(postData));
   addMethod(new CHandle::FALCON_FUNCTION_NAME(getInfo));

   m_afterExec = new CHandle::PStepAfterExec;
}

ClassHandle::~ClassHandle()
{
   delete m_afterExec;
}

void ClassHandle::dispose( void* instance ) const
{
   Mod::CurlHandle* inst = static_cast<Mod::CurlHandle*>(instance);
   delete inst;
}

void* ClassHandle::clone( void* instance ) const
{
   Mod::CurlHandle* other = static_cast<Mod::CurlHandle*>(instance);
   Mod::CurlHandle* cl = new Mod::CurlHandle(*other);
   return cl;
}

void* ClassHandle::createInstance() const
{
   return new Mod::CurlHandle(this);
}

void ClassHandle::gcMarkInstance( void* instance, uint32 mark ) const
{
   Mod::CurlHandle* inst = static_cast<Mod::CurlHandle*>(instance);
   inst->gcMark(mark);
}

bool ClassHandle::gcCheckInstance( void* instance, uint32 mark ) const
{
   Mod::CurlHandle* inst = static_cast<Mod::CurlHandle*>(instance);
   return inst->currentMark() >= mark;
}




namespace CMulti
{
//=======================================================================================================
//
//

static void internal_handle_add( Function* func, VMContext *ctx, Item* i_handle )
{
   ModuleCurl* cmod = static_cast<ModuleCurl*>(func->methodOf()->module());
   Class* clsHandle = cmod->handleClass();

   if( i_handle == 0 || ! i_handle->isInstanceOf( clsHandle ) )
   {
      throw func->paramError();
   }

   Mod::CurlMultiHandle* mh = ctx->tself< Mod::CurlMultiHandle* >();
   Mod::CurlHandle* sh = static_cast< Mod::CurlHandle* >(i_handle->asInst());

   if( ! mh->addHandle(sh) )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_HISIN, __LINE__, SRC)
            .desc( curl_err_easy_already_in ) );
   }
}

/*#
	@class Multi
	@brief Interface to CURL multi_* operations.
	@optparam ... @a Handle instances to be immediately added.

	The Multi interface is meant to perform multiple CURL connections
	handled by a single application.

	A @b Multi instance lifetime is usually like the following:
	- Add one or more pre-configured @a Handle instances.
	- Loop on  @a Multi.perform() up to when it returns 0 indicating that all transfers are complete.

   For example, a minimal operation may be like the following:
   @code
   import from curl
   h1 = curl.Handle( "http://www.falconpl.org" ).setOutString()
   h2 = curl.Handle( "http://www.google.com" ).setOutString()

   hm = curl.Multi( h1, h2 )
   loop
       v = hm.perform()
       > "Currently ", v, " transfers ongoing."
       sleep(0.1)
   end v == 0

   > h1.getData()
   > h2.getData()
   @endcode
*/

FALCON_DECLARE_FUNCTION(init, "...")
FALCON_DEFINE_FUNCTION_P1(init)
{
   for ( int i = 0; i < ctx->paramCount(); ++i )
   {
      internal_handle_add( this, ctx, ctx->param(i) );
   }
   ctx->returnFrame(ctx->self());
}


/*#
   @method add Multi
   @brief Adds an @a Handle instance to the multi interface.
   @param h The @a Handle instance to be added.

   Adds a handle to an existing curl multihandle.
*/
FALCON_DECLARE_FUNCTION(add, "h:Handle")
FALCON_DEFINE_FUNCTION_P1(add)
{
   Item* i_handle = ctx->param(0);
   internal_handle_add( this, ctx, i_handle );
   ctx->returnFrame();
}


/*#
   @method remove Multi
   @brief Adds an @a Handle instance to the multi interface.
   @param h The @a Handle instance to be added.

   Adds a handle to an existing curl multihandle.
*/
FALCON_DECLARE_FUNCTION(remove, "h:Handle")
FALCON_DEFINE_FUNCTION_P1(remove)
{
   ModuleCurl* cmod = static_cast<ModuleCurl*>(methodOf()->module());
   Class* clsHandle = cmod->handleClass();

   Item* i_handle = ctx->param(0);
   if( i_handle == 0 || ! i_handle->isInstanceOf(clsHandle) )
   {
      throw paramError();
   }

   Mod::CurlMultiHandle* mh = ctx->tself< Mod::CurlMultiHandle* >();
   Mod::CurlHandle* sh = static_cast< Mod::CurlHandle* >(i_handle->asInst());


   if( ! mh->removeHandle(sh) )
   {
      throw new CurlError( ErrorParam( FALCON_ERROR_CURL_HNOIN, __LINE__, SRC )
            .desc( curl_err_easy_not_in ) );
   }
   ctx->returnFrame();
}


/*#
   @method perform Multi
   @brief Starts or proceeds with the transfers.
   @return The count of remaining operations to be handled.

   The calling application should call repeatedly this method
   until it returns 0, indicating that all the transfers are
   compelete.
*/
FALCON_DECLARE_FUNCTION(perform, "")
FALCON_DEFINE_FUNCTION_P1(perform)
{
   Mod::CurlMultiHandle* mh = ctx->tself< Mod::CurlMultiHandle* >();

   int rh = 0;
   CURLMcode ret;
   do{
      ret = curl_multi_perform( mh->handle(), &rh );
   }
   while ( ret == CURLM_CALL_MULTI_PERFORM );

   if ( ret != CURLM_OK )
   {
      throw_merror( FALCON_ERROR_CURL_MULTI, __LINE__, curl_err_multi_error, ret );
   }
   ctx->returnFrame( (int64) rh );
}
}


ClassMulti::ClassMulti():
         Class("Multi")
{
   setConstuctor(new CMulti::FALCON_FUNCTION_NAME(init));

   addMethod(new CMulti::FALCON_FUNCTION_NAME(add));
   addMethod(new CMulti::FALCON_FUNCTION_NAME(remove));
   addMethod(new CMulti::FALCON_FUNCTION_NAME(perform));
}

ClassMulti::~ClassMulti()
{

}

void ClassMulti::dispose( void* instance ) const
{
   Mod::CurlMultiHandle* inst = static_cast<Mod::CurlMultiHandle*>(instance);
   delete inst;
}

void* ClassMulti::clone( void* instance ) const
{
   Mod::CurlMultiHandle* other = static_cast<Mod::CurlMultiHandle*>(instance);
   return new Mod::CurlMultiHandle(*other);
}

void* ClassMulti::createInstance() const
{
   return new Mod::CurlMultiHandle;
}

void ClassMulti::gcMarkInstance( void* instance, uint32 mark ) const
{
   Mod::CurlMultiHandle* inst = static_cast<Mod::CurlMultiHandle*>(instance);
   inst->gcMark(mark);
}

bool ClassMulti::gcCheckInstance( void* instance, uint32 mark ) const
{
   Mod::CurlMultiHandle* inst = static_cast<Mod::CurlMultiHandle*>(instance);
   return inst->currentMark() >= mark;
}


/*#
   @class CurlError
   @brief Error generated by cURL while operating.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   See the Error class in the core module.
*/


/*#
   @function dlaod
   @brief Downloads file.
   @param uri The uri to be downloaded.
   @optparam stream a Stream where to download the data.

   Downloads a file from a remote source and stores it on a string,
   or on a @b stream, as in the following sequence:
   @code
      import from curl in curl

      data = curl.Handle( "http://www.falconpl.org" ).setOutString().exec().getData()
      // equivalent:
      data = curl.dload( "http://www.falconpl.org" )
   @endcode
*/
FALCON_DEFINE_FUNCTION_P1(dload)
{
   static Class* ctxURI = Engine::instance()->stdHandlers()->uriClass();
   static Class* ctxStream = Engine::instance()->stdHandlers()->streamClass();

   Item* i_uri = ctx->param(0);
   Item* i_stream = ctx->param(1);

   if ( i_uri == 0 || ! (i_uri->isString() || i_uri->isInstanceOf(ctxURI))
         || (i_stream != 0 && ! (i_stream->isNil() || i_stream->isInstanceOf(ctxStream) ) ) )
   {
      throw paramError();
   }

   ModuleCurl* mc = static_cast<ModuleCurl*>(module());
   Mod::CurlHandle* ca = new Mod::CurlHandle(mc->handleClass());

   internal_curl_init( this, ctx, ca, i_uri );

   if( i_stream == 0 || i_stream->isNil() )
   {
      ca->setOnDataGetString();
   }
   else
   {
      ca->setOnDataStream( static_cast<Stream*>(i_stream->asInst()) );
   }

  CURLcode retval = curl_easy_perform(ca->handle());
  if( retval != CURLE_OK )
  {
     ca->cleanup();
     ca->gcMark(1); // let the gc kill it
     throw_error( FALCON_ERROR_CURL_EXEC, __LINE__, curl_err_exec, retval );
  }

  ca->cleanup();
  delete ca;

  if( i_stream == 0 || i_stream->isNil() )
  {
     ctx->returnFrame( FALCON_GC_HANDLE(ca->getData()) );
  }
  else {
     ctx->returnFrame();
  }
}



/*#
   @method version CURL
   @brief Returns the version of libcurl
   @return A string containing the description of the version of the
   cURL library being in use.
*/
FALCON_DECLARE_FUNCTION(version, "");
FALCON_DEFINE_FUNCTION_P1(version)
{
   ctx->returnFrame( FALCON_GC_HANDLE(new String( ::curl_version() ) ) );
}


//=======================================================================
// ClassCURL
//=======================================================================
/*# @class CURL
  @brief Generic functions and enumerations

  Static enumeration members of this class:
  - READFUNC_ABORT: value that can be returned by read callback function to abort the read
  - READFUNC_PAUSE: value that can be returned by read callback function to pause the read
  - WRITEFUNC_PAUSE: value that can be returned by write callback function to pause the read
 */
ClassCURL::ClassCURL():
         Class("CURL")
{
   addConstant("READFUNC_ABORT",(int64) CURL_READFUNC_ABORT );
   addConstant("READFUNC_PAUSE",(int64) CURL_READFUNC_PAUSE );
   addConstant("WRITEFUNC_PAUSE",(int64) CURL_WRITEFUNC_PAUSE );

   addMethod(new FALCON_FUNCTION_NAME(version), true);
}


ClassCURL::~ClassCURL()
{}

void ClassCURL::dispose( void* ) const
{
   // nothing to do
}

void* ClassCURL::clone( void* ) const
{
   return 0;
}

void* ClassCURL::createInstance() const
{
   return 0;
}

//=======================================================================
// ClassOPT
//=======================================================================

ClassOPT::ClassOPT():
         Class("OPT")
{
   addConstant("VERBOSE",(int64) CURLOPT_VERBOSE );
   addConstant("HEADER",(int64) CURLOPT_HEADER );
   addConstant("NOPROGRESS",(int64) CURLOPT_NOPROGRESS );
   addConstant("HTTPPROXYTUNNEL",(int64) CURLOPT_HTTPPROXYTUNNEL );
#if LIBCURL_VERSION_NUM >= 0x071904
   addConstant("HTTPPROXYTUNNEL",(int64) CURLOPT_HTTPPROXYTUNNEL );
#endif
   addConstant("TCP_NODELAY",(int64) CURLOPT_TCP_NODELAY );
   addConstant("AUTOREFERER",(int64) CURLOPT_AUTOREFERER );
   addConstant("FOLLOWLOCATION",(int64) CURLOPT_FOLLOWLOCATION );
   addConstant("UNRESTRICTED_AUTH",(int64) CURLOPT_UNRESTRICTED_AUTH );
   addConstant("PUT",(int64) CURLOPT_PUT );
   addConstant("POST",(int64) CURLOPT_POST );
   addConstant("COOKIESESSION",(int64) CURLOPT_COOKIESESSION );
   addConstant("HTTPGET",(int64) CURLOPT_HTTPGET );
   addConstant("IGNORE_CONTENT_LENGTH",(int64) CURLOPT_IGNORE_CONTENT_LENGTH );
   addConstant("DIRLISTONLY",(int64) CURLOPT_DIRLISTONLY );
   addConstant("APPEND",(int64) CURLOPT_APPEND );
   addConstant("FTP_USE_EPRT",(int64) CURLOPT_FTP_USE_EPRT );
   addConstant("FTP_USE_EPSV",(int64) CURLOPT_FTP_USE_EPSV );
   addConstant("FTP_CREATE_MISSING_DIRS",(int64) CURLOPT_FTP_CREATE_MISSING_DIRS );
   addConstant("CRLF",(int64) CURLOPT_CRLF );
   addConstant("FILETIME",(int64) CURLOPT_FILETIME );
   addConstant("NOBODY",(int64) CURLOPT_NOBODY );
   addConstant("UPLOAD",(int64) CURLOPT_UPLOAD );
   addConstant("FORBID_REUSE",(int64) CURLOPT_FORBID_REUSE );
   addConstant("FRESH_CONNECT",(int64) CURLOPT_FRESH_CONNECT );

   addConstant("CONNECT_ONLY",(int64) CURLOPT_CONNECT_ONLY);
   addConstant("SSLENGINE_DEFAULT",(int64) CURLOPT_SSLENGINE_DEFAULT);
   addConstant("SSL_VERIFYPEER",(int64) CURLOPT_SSL_VERIFYPEER);
#if LIBCURL_VERSION_NUM >= 0x071901
   addConstant("CERTINFO",(int64) CURLOPT_CERTINFO);
#endif
   addConstant("SSL_VERIFYHOST",(int64) CURLOPT_SSL_VERIFYHOST);
   addConstant("SSL_SESSIONID_CACHE",(int64) CURLOPT_SSL_SESSIONID_CACHE);

#if CURLOPT_PROTOCOLS
   addConstant("PROTOCOLS",(int64) CURLOPT_PROTOCOLS);
   addConstant("REDIR_PROTOCOLS",(int64) CURLOPT_REDIR_PROTOCOLS);
#endif
   addConstant("PROXYPORT",(int64) CURLOPT_PROXYPORT);
   addConstant("PROXYTYPE",(int64) CURLOPT_PROXYTYPE);

   addConstant("LOCALPORT",(int64) CURLOPT_LOCALPORT);
   addConstant("LOCALPORTRANGE",(int64) CURLOPT_LOCALPORTRANGE);
   addConstant("DNS_CACHE_TIMEOUT",(int64) CURLOPT_DNS_CACHE_TIMEOUT);
   addConstant("DNS_USE_GLOBAL_CACHE",(int64) CURLOPT_DNS_USE_GLOBAL_CACHE);
   addConstant("BUFFERSIZE",(int64) CURLOPT_BUFFERSIZE);
   addConstant("PORT",(int64) CURLOPT_PORT);
#if LIBCURL_VERSION_NUM >= 0x071900
   addConstant("ADDRESS_SCOPE",(int64) CURLOPT_ADDRESS_SCOPE);
#endif
   addConstant("NETRC",(int64) CURLOPT_NETRC);

      addConstant("HTTPAUTH",(int64) CURLOPT_HTTPAUTH);

      addConstant("PROXYAUTH",(int64) CURLOPT_PROXYAUTH);
      addConstant("MAXREDIRS",(int64) CURLOPT_MAXREDIRS);
   #if LIBCURL_VERSION_NUM >= 0x071901
      addConstant("POSTREDIR",(int64) CURLOPT_POSTREDIR);
   #endif
      addConstant("HTTP_VERSION",(int64) CURLOPT_HTTP_VERSION);

      addConstant("HTTP_CONTENT_DECODING",(int64) CURLOPT_HTTP_CONTENT_DECODING);
      addConstant("HTTP_TRANSFER_DECODING",(int64) CURLOPT_HTTP_TRANSFER_DECODING);
   #if LIBCURL_VERSION_NUM >= 0x071904
      addConstant("TFTP_BLKSIZE",(int64) CURLOPT_TFTP_BLKSIZE);
   #endif
      addConstant("FTP_RESPONSE_TIMEOUT",(int64) CURLOPT_FTP_RESPONSE_TIMEOUT);
      addConstant("USE_SSL",(int64) CURLOPT_USE_SSL);

      addConstant("FTPSSLAUTH",(int64) CURLOPT_FTPSSLAUTH);

      addConstant("FTP_SSL_CCC",(int64) CURLOPT_FTP_SSL_CCC);

      addConstant("NONE",(int64) CURLFTPSSL_CCC_NONE);
      addConstant("PASSIVE",(int64) CURLFTPSSL_CCC_PASSIVE);
      addConstant("ACTIVE",(int64) CURLFTPSSL_CCC_ACTIVE);
      addConstant("FTP_FILEMETHOD",(int64) CURLOPT_FTP_FILEMETHOD);

      addConstant("RESUME_FROM",(int64) CURLOPT_RESUME_FROM);
      addConstant("INFILESIZE",(int64) CURLOPT_INFILESIZE);
      addConstant("MAXFILESIZE",(int64) CURLOPT_MAXFILESIZE);
      addConstant("TIMEVALUE",(int64) CURLOPT_TIMEVALUE);
      addConstant("TIMEOUT",(int64) CURLOPT_TIMEOUT);
      addConstant("TIMEOUT_MS",(int64) CURLOPT_TIMEOUT_MS);
      addConstant("LOW_SPEED_LIMIT",(int64) CURLOPT_LOW_SPEED_LIMIT);
      addConstant("LOW_SPEED_TIME",(int64) CURLOPT_LOW_SPEED_TIME);
      addConstant("MAXCONNECTS",(int64) CURLOPT_MAXCONNECTS);
      addConstant("CONNECTTIMEOUT",(int64) CURLOPT_CONNECTTIMEOUT);
      addConstant("CONNECTTIMEOUT_MS",(int64) CURLOPT_CONNECTTIMEOUT_MS);
      addConstant("IPRESOLVE",(int64) CURLOPT_IPRESOLVE);

      addConstant("SSLVERSION",(int64) CURLOPT_SSLVERSION);

      addConstant("SSH_AUTH_TYPES",(int64) CURLOPT_SSH_AUTH_TYPES);

      addConstant("NEW_FILE_PERMS",(int64) CURLOPT_NEW_FILE_PERMS);
      addConstant("NEW_DIRECTORY_PERMS",(int64) CURLOPT_NEW_DIRECTORY_PERMS);

      addConstant("RESUME_FROM_LARGE",(int64) CURLOPT_RESUME_FROM_LARGE);
      addConstant("INFILESIZE_LARGE",(int64) CURLOPT_INFILESIZE_LARGE);
      addConstant("MAXFILESIZE_LARGE",(int64) CURLOPT_MAXFILESIZE_LARGE);
      addConstant("MAX_SEND_SPEED_LARGE",(int64) CURLOPT_MAX_SEND_SPEED_LARGE);
      addConstant("MAX_RECV_SPEED_LARGE",(int64) CURLOPT_MAX_RECV_SPEED_LARGE);

      addConstant("URL",(int64) CURLOPT_URL);
      addConstant("PROXY",(int64) CURLOPT_PROXY);
   #if LIBCURL_VERSION_NUM >= 0x071904
      addConstant("NOPROXY",(int64) CURLOPT_NOPROXY);
      addConstant("SOCKS5_GSSAPI_SERVICE",(int64) CURLOPT_SOCKS5_GSSAPI_SERVICE);
   #endif
      addConstant("INTERFACE",(int64) CURLOPT_INTERFACE);
      addConstant("NETRC_FILE",(int64) CURLOPT_NETRC_FILE);
      addConstant("USERPWD",(int64) CURLOPT_USERPWD);
      addConstant("PROXYUSERPWD",(int64) CURLOPT_PROXYUSERPWD);
   #if LIBCURL_VERSION_NUM >= 0x071901
      addConstant("USERNAME",(int64) CURLOPT_USERNAME);
      addConstant("PASSWORD",(int64) CURLOPT_PASSWORD);
      addConstant("PROXYUSERNAME",(int64) CURLOPT_PROXYUSERNAME);
      addConstant("PROXYPASSWORD",(int64) CURLOPT_PROXYPASSWORD);
   #endif
      addConstant("ENCODING",(int64) CURLOPT_ENCODING);
      addConstant("REFERER",(int64) CURLOPT_REFERER);
      addConstant("USERAGENT",(int64) CURLOPT_USERAGENT);
      addConstant("COOKIE",(int64) CURLOPT_COOKIE);
      addConstant("COOKIEFILE",(int64) CURLOPT_COOKIEFILE);
      addConstant("COOKIEJAR",(int64) CURLOPT_COOKIEJAR);
      addConstant("COOKIELIST",(int64) CURLOPT_COOKIELIST);
      addConstant("FTPPORT",(int64) CURLOPT_FTPPORT);
      addConstant("FTP_ALTERNATIVE_TO_USER",(int64) CURLOPT_FTP_ALTERNATIVE_TO_USER);
      addConstant("FTP_ACCOUNT",(int64) CURLOPT_FTP_ACCOUNT);
      addConstant("RANGE",(int64) CURLOPT_RANGE);
      addConstant("CUSTOMREQUEST",(int64) CURLOPT_CUSTOMREQUEST);
      addConstant("SSLCERT",(int64) CURLOPT_SSLCERT);
      addConstant("SSLCERTTYPE",(int64) CURLOPT_SSLCERTTYPE);
      addConstant("SSLKEY",(int64) CURLOPT_SSLKEY);
      addConstant("SSLKEYTYPE",(int64) CURLOPT_SSLKEYTYPE);
      addConstant("KEYPASSWD",(int64) CURLOPT_KEYPASSWD);
      addConstant("SSLENGINE",(int64) CURLOPT_SSLENGINE);
      addConstant("CAINFO",(int64) CURLOPT_CAINFO);
   #if LIBCURL_VERSION_NUM >= 0x071900
      addConstant("ISSUERCERT",(int64) CURLOPT_ISSUERCERT);
      addConstant("CRLFILE",(int64) CURLOPT_CRLFILE);
   #endif
      addConstant("CAPATH",(int64) CURLOPT_CAPATH);
      addConstant("RANDOM_FILE",(int64) CURLOPT_RANDOM_FILE);
      addConstant("EGDSOCKET",(int64) CURLOPT_EGDSOCKET);
      addConstant("SSL_CIPHER_LIST",(int64) CURLOPT_SSL_CIPHER_LIST);
      addConstant("KRBLEVEL",(int64) CURLOPT_KRBLEVEL);
      addConstant("SSH_HOST_PUBLIC_KEY_MD5",(int64) CURLOPT_SSH_HOST_PUBLIC_KEY_MD5);
      addConstant("SSH_PUBLIC_KEYFILE",(int64) CURLOPT_SSH_PUBLIC_KEYFILE);
      addConstant("SSH_PRIVATE_KEYFILE",(int64) CURLOPT_SSH_PRIVATE_KEYFILE);

   #ifdef CURLOPT_SSH_KNOWNHOSTS
      addConstant("SSH_KNOWNHOSTS",(int64) CURLOPT_SSH_KNOWNHOSTS);
   #endif

      // List options
      addConstant("HTTPHEADER",(int64) CURLOPT_HTTPHEADER);
      addConstant("HTTP200ALIASES",(int64) CURLOPT_HTTP200ALIASES);
      addConstant("QUOTE",(int64) CURLOPT_QUOTE);
      addConstant("POSTQUOTE",(int64) CURLOPT_POSTQUOTE);
      addConstant("PREQUOTE",(int64) CURLOPT_PREQUOTE);

      // To be implemented separately
      /*
      CURLOPT_HTTPPOST

      CURLOPT_SSH_KEYFUNCTION
      CURLOPT_SSH_KEYDATA

      CURLOPT_SHARE (?)

      CURLOPT_TELNETOPTIONS
      CURLOPT_TIMECONDITION
      */

}


//=======================================================================
// ClassOPT
//=======================================================================

ClassPROXY::ClassPROXY():
         Class("PROXY")
{
   addConstant("HTTP",(int64) CURLPROXY_HTTP);
#if LIBCURL_VERSION_NUM >= 0x071904
   addConstant("HTTP_1_0",(int64) CURLPROXY_HTTP_1_0);
#endif
   addConstant("SOCKS4",(int64) CURLPROXY_SOCKS4);
   addConstant("SOCKS5",(int64) CURLPROXY_SOCKS5);
   addConstant("SOCKS4A",(int64) CURLPROXY_SOCKS4A);
}


//=======================================================================
// ClassNETRC
//=======================================================================

ClassNETRC::ClassNETRC():
         Class("NETRC")
{
   addConstant("OPTIONAL",(int64) CURL_NETRC_OPTIONAL);
   addConstant("IGNORED",(int64) CURL_NETRC_IGNORED);
}

//=======================================================================
// ClassAUTH
//=======================================================================

ClassAUTH::ClassAUTH():
         Class("AUTH")
{
   addConstant("BASIC",(int64) CURLAUTH_BASIC);
   addConstant("DIGEST",(int64) CURLAUTH_DIGEST);
#if LIBCURL_VERSION_NUM >= 0x071903
   addConstant("DIGEST_IE",(int64) CURLAUTH_DIGEST_IE);
#endif
   addConstant("GSSNEGOTIATE",(int64) CURLAUTH_GSSNEGOTIATE);
   addConstant("NTLM",(int64) CURLAUTH_NTLM);
   addConstant("ANY",(int64) CURLAUTH_ANY);
   addConstant("ANYSAFE",(int64) CURLAUTH_ANYSAFE);
}

//=======================================================================
// ClassHTTP
//=======================================================================

ClassHTTP::ClassHTTP():
         Class("HTTP")
{
   addConstant("VERSION_NONE",(int64) CURL_HTTP_VERSION_NONE);
   addConstant("VERSION_1_0",(int64) CURL_HTTP_VERSION_1_0);
   addConstant("VERSION_1_1",(int64) CURL_HTTP_VERSION_1_1);
}

//=======================================================================
// ClassUSESSL
//=======================================================================

ClassUSESSL::ClassUSESSL():
         Class("USESSL")
{
   addConstant("NONE",(int64) CURLUSESSL_NONE);
   addConstant("TRY",(int64) CURLUSESSL_TRY);
   addConstant("CONTROL",(int64) CURLUSESSL_CONTROL);
   addConstant("ALL",(int64) CURLUSESSL_ALL);
}


//=======================================================================
// Class ClassFTPAUTH
//=======================================================================

ClassFTPAUTH::ClassFTPAUTH():
         Class("FTPAUTH")
{
   addConstant("DEFAULT",(int64) CURLFTPAUTH_DEFAULT);
   addConstant("SSL",(int64) CURLFTPAUTH_SSL);
   addConstant("TLS",(int64) CURLFTPAUTH_TLS);
}


//=======================================================================
// Class SSL_CCC
//=======================================================================

ClassSSL_CCC::ClassSSL_CCC():
         Class("SSL_CCC")
{
   addConstant("NONE",(int64) CURLFTPSSL_CCC_NONE);
   addConstant("PASSIVE",(int64) CURLFTPSSL_CCC_PASSIVE);
   addConstant("ACTIVE",(int64) CURLFTPSSL_CCC_ACTIVE);
}


//=======================================================================
// Class FTPMETHOD
//=======================================================================

ClassFTPMETHOD::ClassFTPMETHOD():
         Class("FTPMETHOD")
{
   addConstant("MULTICWD",(int64) CURLFTPMETHOD_MULTICWD);
   addConstant("NOCWD",(int64) CURLFTPSSL_CCC_PASSIVE);
   addConstant("SINGLECWD",(int64) CURLFTPMETHOD_SINGLECWD);
}

//=======================================================================
// Class FTPMETHOD
//=======================================================================

ClassIPRESOLVE::ClassIPRESOLVE():
         Class("IPRESOLVE")
{
   addConstant("WHATEVER",(int64) CURL_IPRESOLVE_WHATEVER);
   addConstant("V4",(int64) CURL_IPRESOLVE_V4);
   addConstant("V6",(int64) CURL_IPRESOLVE_V6);
}

//=======================================================================
// Class FTPMETHOD
//=======================================================================

ClassSSLVERSION::ClassSSLVERSION():
         Class("SSLVERSION")
{
   addConstant("DEFAULT",(int64) CURL_SSLVERSION_DEFAULT);
   addConstant("TLSv1",(int64) CURL_SSLVERSION_TLSv1);
   addConstant("SSLv2",(int64) CURL_SSLVERSION_SSLv2);
   addConstant("SSLv3",(int64) CURL_SSLVERSION_SSLv3);
}

//=======================================================================
// Class FTPMETHOD
//=======================================================================

ClassSSH_AUTH::ClassSSH_AUTH():
         Class("SSH_AUTH")
{
   addConstant("PUBLICKEY",(int64) CURLSSH_AUTH_PUBLICKEY);
   addConstant("PASSWORD",(int64) CURLSSH_AUTH_PASSWORD);
   addConstant("HOST",(int64) CURLSSH_AUTH_HOST);
   addConstant("KEYBOARD",(int64) CURLSSH_AUTH_KEYBOARD);
   addConstant("ANY",(int64) CURLSSH_AUTH_ANY);
}

//=======================================================================
// ClassINFO
//=======================================================================

ClassINFO::ClassINFO():
         Class("INFO")
{
   addConstant("EFFECTIVE_URL",(int64) CURLINFO_EFFECTIVE_URL);
   addConstant("RESPONSE_CODE",(int64) CURLINFO_RESPONSE_CODE);
   addConstant("HTTP_CONNECTCODE",(int64) CURLINFO_HTTP_CONNECTCODE);
   addConstant("FILETIME",(int64) CURLINFO_FILETIME);
   addConstant("TOTAL_TIME",(int64) CURLINFO_TOTAL_TIME);
   addConstant("NAMELOOKUP_TIME",(int64) CURLINFO_NAMELOOKUP_TIME);
   addConstant("CONNECT_TIME",(int64) CURLINFO_CONNECT_TIME);
   #if LIBCURL_VERSION_NUM >= 0x071900
   addConstant("APPCONNECT_TIME",(int64) CURLINFO_APPCONNECT_TIME);
   #endif
   addConstant("PRETRANSFER_TIME",(int64) CURLINFO_PRETRANSFER_TIME);
   addConstant("STARTTRANSFER_TIME",(int64) CURLINFO_STARTTRANSFER_TIME);
   addConstant("REDIRECT_TIME",(int64) CURLINFO_REDIRECT_TIME);
   addConstant("REDIRECT_COUNT",(int64) CURLINFO_REDIRECT_COUNT);
   addConstant("REDIRECT_URL",(int64) CURLINFO_REDIRECT_URL);
   addConstant("SIZE_UPLOAD",(int64) CURLINFO_SIZE_UPLOAD);
   addConstant("SIZE_DOWNLOAD",(int64) CURLINFO_SIZE_DOWNLOAD);
   addConstant("SPEED_DOWNLOAD",(int64) CURLINFO_SPEED_DOWNLOAD);
   addConstant("SPEED_UPLOAD",(int64) CURLINFO_SPEED_UPLOAD);
   addConstant("HEADER_SIZE",(int64) CURLINFO_HEADER_SIZE);
   addConstant("REQUEST_SIZE",(int64) CURLINFO_REQUEST_SIZE);
   addConstant("SSL_VERIFYRESULT",(int64) CURLINFO_SSL_VERIFYRESULT);
   addConstant("SSL_ENGINES",(int64) CURLINFO_SSL_ENGINES);
   addConstant("CONTENT_LENGTH_DOWNLOAD",(int64) CURLINFO_CONTENT_LENGTH_DOWNLOAD);
   addConstant("CONTENT_LENGTH_UPLOAD",(int64) CURLINFO_CONTENT_LENGTH_UPLOAD);
   addConstant("CONTENT_TYPE",(int64) CURLINFO_CONTENT_TYPE);
   addConstant("HTTPAUTH_AVAIL",(int64) CURLINFO_HTTPAUTH_AVAIL);
   addConstant("PROXYAUTH_AVAIL",(int64) CURLINFO_PROXYAUTH_AVAIL);
   addConstant("NUM_CONNECTS",(int64) CURLINFO_NUM_CONNECTS);
   #if LIBCURL_VERSION_NUM >= 0x071900
   addConstant("PRIMARY_IP",(int64) CURLINFO_PRIMARY_IP);
   #endif
   addConstant("COOKIELIST",(int64) CURLINFO_COOKIELIST);
   addConstant("FTP_ENTRY_PATH",(int64) CURLINFO_FTP_ENTRY_PATH);
   addConstant("SSL_ENGINES",(int64) CURLINFO_SSL_ENGINES);
   #if LIBCURL_VERSION_NUM >= 0x071904
   addConstant("CONDITION_UNMET",(int64) CURLINFO_CONDITION_UNMET);
   #endif

   /**
    * Separately handled
    *    CURLINFO_PRIVATE -> CURLOPT_PRIVATE
    *    CURLINFO_LASTSOCKET -> socket?
    *    CURLINFO_CERTINFO
    */
}



}
}

/* end of curl_mod.cpp */
