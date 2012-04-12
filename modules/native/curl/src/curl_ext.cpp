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

#include <falcon/engine.h>
#include <curl/curl.h>

#include "curl_mod.h"
#include "curl_ext.h"
#include "curl_st.h"


/*# @beginmodule curl
*/

namespace Falcon {
namespace Ext {

static void throw_error( int code, int line, const String& cd, const CURLcode retval )
{
   String error = String( curl_easy_strerror( retval ) );
   throw new Mod::CurlError( ErrorParam( code, line )
         .desc( cd )
         .extra( error.A(" (").N(retval).A(")") )
         );
}

static void throw_merror( int code, int line, const String& cd, const CURLMcode retval )
{
   String error = String( curl_multi_strerror( retval ) );
   throw new Mod::CurlError( ErrorParam( code, line )
         .desc( cd )
         .extra( error.A(" (").N(retval).A(")") )
         );
}

static void internal_curl_init( VMachine* vm, Mod::CurlHandle* h, Item* i_uri )
{
   CURL* curl = h->handle();

   // we had a general init error from curl
   if ( curl == 0 )
   {
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_INIT, __LINE__ )
                  .desc( FAL_STR( curl_err_init ) )
                  .extra( FAL_STR( curl_err_resources ) ));
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
   else if( i_uri->isOfClass( "URI" ) )
   {
      URI* uri = (URI*) i_uri->asObjectSafe()->getUserData();
      AutoCString curi( uri->get(true) );

      retval = curl_easy_setopt( curl, CURLOPT_URL, curi.c_str() );
   }
   else
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "[S|URI]" ) );
   }

   if( retval != CURLE_OK )
   {
      throw_error( FALCON_ERROR_CURL_INIT, __LINE__, FAL_STR( curl_err_init ), retval );
   }
}

/*#
   @function curl_version
   @brief Returns the version of libcurl
   @return A string containing the description of the used version-.
*/

FALCON_FUNC  curl_version( ::Falcon::VMachine *vm )
{
   vm->retval( new CoreString( ::curl_version() ) );
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

FALCON_FUNC  Handle_init( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );
   Item* i_uri = vm->param(0);

   internal_curl_init( vm, h, i_uri );
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

FALCON_FUNC  Handle_exec( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );
   CURL* curl = h->handle();

   if ( curl == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   CURLcode retval = curl_easy_perform(curl);
   if( retval != CURLE_OK )
   {
      throw_error( FALCON_ERROR_CURL_EXEC, __LINE__, FAL_STR( curl_err_exec ), retval );
   }
   vm->retval( vm->self() );
}

/*#
   @method setOutConsole Handle
   @brief Asks for subsequent transfer(s) to be sent to process console (raw stdout).
   @return self (to put this call in a chain)

   This is the default at object creation.
*/
FALCON_FUNC  Handle_setOutConsole( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   h->setOnDataStdOut();
}

/*#
   @method setOutString Handle
   @brief Asks for subsequent transfer(s) to be stored in a temporary string.
   @return self (to put this call in a chain)


   After @a Handle.exec has been called, the data will be available in
   a string that can be retrieved via the @a Handle.getData method.

*/

FALCON_FUNC  Handle_setOutString( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   h->setOnDataGetString();
   vm->retval( vm->self() );
}


/*#
   @method setOutStream Handle
   @brief Asks for subsequent transfer(s) to be stored in a given stream.
   @param stream The stream to be used.
   @return self (to put this call in a chain)

   When called, @a Handle.exec will store incoming data in this stream object
   via binary Stream.write operations.
*/

FALCON_FUNC  Handle_setOutStream( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   Item* i_stream = vm->param(0);

   if ( i_stream == 0 || ! i_stream->isOfClass("Stream") )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "Stream" ) );
   }

   h->setOnDataStream( (Stream*) i_stream->asObjectSafe()->getUserData() );
   vm->retval( vm->self() );
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
FALCON_FUNC  Handle_setOutCallback( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   Item* i_cb = vm->param(0);

   if ( i_cb  == 0 || ! i_cb->isCallable() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "C" ) );
   }

   h->setOnDataCallback( *i_cb );
   vm->retval( vm->self() );
}

// not yet active
/*
   @method setOutMessage Handle
   @brief Asks for subsequent transfer(s) to be handled as a message broadcast.
   @param msg A string representing a message or a VMSlot.
   @return self (to put this call in a chain)

   This method instructs this handle to perform message broadcast when data
   is received.

   When called, @a Handle.exec will repeatedly broadcast @msg sending two parameters:
   itself (this Handle object) and the received data, as a binary string.

   The string is not encoded in any format, and could be considered filled with binary
   data.
   vm->retval( vm->self() );
*/
/*
FALCON_FUNC  Handle_setOutMessage( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   Item* i_msg = vm->param(0);

   if ( i_msg  == 0 ||
         (! i_msg->isString() && ! i_msg->isOfClass("VMSlot") ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "S|VMSlot" ) );
   }

   if( i_msg->isString() )
   {
      h->setOnDataMessage( *i_msg->asString() );
   }
   else
   {
      CoreSlot* cs = (CoreSlot*) i_msg->asObjectSafe()->getUserData();
      h->setOnDataMessage( cs->name() );
   }
}
*/

/*#
   @method cleanup Handle
   @brief Close a connection and destroys all associated data.

   After this call, the handle is not usable anymore.
   This is executed also automatically at garbage collection, but
   the user may be interested in clearing the data as soon as possible.
*/
FALCON_FUNC  Handle_cleanup( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   h->cleanup();
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
FALCON_FUNC  Handle_setInCallback( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   Item* i_cb = vm->param(0);

   if ( i_cb  == 0 || ! i_cb->isCallable() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "C" ) );
   }

   h->setReadCallback( *i_cb );
   vm->retval( vm->self() );
}


/*#
   @method setInStream Handle
   @brief Asks for subsequent upload(s) to read data from the given stream.
   @param stream The stream to be used.
   @return self (to put this call in a chain)

   When called, @a Handle.exec will read data to be uploaded from this
   stream.
*/

FALCON_FUNC  Handle_setInStream( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   Item* i_stream = vm->param(0);

   if ( i_stream == 0 || ! i_stream->isOfClass("Stream") )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__)
            .extra( "Stream" ) );
   }

   h->setReadStream( (Stream*) i_stream->asObjectSafe()->getUserData() );
   vm->retval( vm->self() );
}

/*#
   @method getData Handle
   @brief Gets data temporarily stored in a string during a transfer.
   @return A string containing data that has been transfered.

   This function returns the data received in the meanwhile. This data
   is captured when the @a Handle.setOutString option has been set.

*/
FALCON_FUNC  Handle_getData( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   CoreString* s = h->getData();
   if( s != 0 )
   {
      vm->retval( s );
   }
}


static void internal_setOpt( VMachine* vm, Mod::CurlHandle* h, CURLoption iOpt, Item* i_data )
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
   case CURLOPT_SSL_VERIFYHOST:
   case CURLOPT_SSL_SESSIONID_CACHE:
      {
         long bVal = i_data->isTrue() ? 1 : 0;
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
             throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                   .extra( FAL_STR( curl_err_setopt ) ));
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
              throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                    .extra( FAL_STR( curl_err_setopt ) ));
           }

           AutoCString cstr( *i_data );
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
              throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                    .extra( FAL_STR( curl_err_setopt ) ));
          }

          CoreArray* items = i_data->asArray();
          struct curl_slist *slist = h->slistFromArray( items );
          if ( slist == 0 )
          {
               throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                     .extra( FAL_STR( curl_err_setopt ) ));
          }

          ret = curl_easy_setopt( curl, iOpt, slist );
       }
       break;

    default:
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
             .extra( FAL_STR( curl_err_unkopt ) ));
   }

   if( ret != CURLE_OK )
   {
      throw_error( FALCON_ERROR_CURL_SETOPT, __LINE__, FAL_STR( curl_err_setopt ), ret );
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

FALCON_FUNC  Handle_setOption( ::Falcon::VMachine *vm )
{
   Item* i_option = vm->param(0);
   Item* i_data = vm->param(1);

   if ( i_option == 0 || ! i_option->isInteger()
         || i_data == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "I,X" ) );
   }

   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   CURLoption iOpt = (CURLoption) i_option->asInteger();
   internal_setOpt( vm, h, iOpt, i_data );
   vm->retval( vm->self() );
}


/*#
   @method setOptions Handle
   @brief Sets a list of cURL option for this specific handle.
   @param opts A dictionary of options, where each key is an option number, and its value is the option value.
   @return self (to put this call in a chain)
*/

FALCON_FUNC  Handle_setOptions( ::Falcon::VMachine *vm )
{
   Item* i_opts = vm->param(0);

   if ( i_opts == 0 || ! i_opts->isDict() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "D" ) );
   }

   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   Iterator iter( &i_opts->asDict()->items() );
   while( iter.hasCurrent() )
   {
      Item& opt = iter.getCurrentKey();
      if( ! opt.isInteger() )
      {
         throw new ParamError( ErrorParam( e_param_type, __LINE__ )
                     .extra( "D[I=>X]" ) );
      }

      internal_setOpt( vm, h, (CURLoption) opt.asInteger(), &iter.getCurrent() );
      iter.next();
   }
   vm->retval( vm->self() );
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

FALCON_FUNC  Handle_postData( ::Falcon::VMachine *vm )
{
   Item* i_data = vm->param(0);

   if ( i_data == 0 || ! i_data->isString() )
   {
     throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
           .extra( "S" ) );
   }

   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
     throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
           .desc( FAL_STR( curl_err_pm ) ) );

   h->postData( *i_data->asString() );
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
FALCON_FUNC  Handle_getInfo( ::Falcon::VMachine *vm )
{
   Item* i_option = vm->param(0);
   if( i_option == 0 || ! i_option->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "N" ) );
   }

   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
     throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
           .desc( FAL_STR( curl_err_pm ) ) );

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
         CoreString* cs = new CoreString();
         cs->bufferize( rv );
         vm->retval( cs );
      }
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
            vm->retval( (int64) rv );
         }
      }
      break;

      // timestamp
   case CURLINFO_FILETIME:
      {
         long rv;
         cerr = curl_easy_getinfo( h->handle(), info, &rv );
         if( cerr == CURLE_OK && rv != -1 )
         {
            time_t trv = (time_t)rv;
            TimeStamp* timestamp = new TimeStamp;
            timestamp->m_timezone = tz_UTC;

            #ifndef FALCON_SYSTEM_WIN
               struct tm rtm;
               struct tm *ftime = gmtime_r( &trv, &rtm );

            #else
               struct tm *ftime = gmtime( &trv );
            #endif

            timestamp->m_year = ftime->tm_year + 1900;
            timestamp->m_month = ftime->tm_mon + 1;
            timestamp->m_day = ftime->tm_mday;
            timestamp->m_hour = ftime->tm_hour;
            timestamp->m_minute = ftime->tm_min;
            timestamp->m_second = ftime->tm_sec;
            timestamp->m_msec = 0;
            Item* i_ts = vm->findGlobalItem("TimeStamp");
            fassert( i_ts->isClass() );
            vm->retval( i_ts->asClass()->createInstance(timestamp) );
         }
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
            vm->retval( (numeric) rv );
         }
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
            CoreArray* ca = new CoreArray;
            curl_slist* p = rv;
            while( p != 0 )
            {
               ca->append( new CoreString( p->data, -1 ) );
               p = p->next;
            }

            curl_slist_free_all( rv );
            vm->retval( ca );
         }
      }
      break;

   default:
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ) );
   }

   if( cerr != CURLE_OK )
   {
      throw_error( FALCON_ERROR_CURL_GETINFO, __LINE__, FAL_STR( curl_err_getinfo ), cerr );
   }
}

/*#
   @function dlaod
   @brief Downloads file.
   @param uri The uri to be downloaded.
   @optparam stream a Stream where to download the data.

   Downloads a file from a remote source and stores it on a string,
   or on a @b stream, as in the following sequence:
   @code
      import from curl

      data = curl.Handle( "http://www.falconpl.org" ).setOutString().exec().getData()
      // equivalent:
      data = curl.dload( "http://www.falconpl.org" )
   @endcode
*/

FALCON_FUNC  curl_dload( ::Falcon::VMachine *vm )
{
   Item* i_uri = vm->param(0);
   Item* i_stream = vm->param(1);

   if ( i_uri == 0 || ! (i_uri->isString() || i_uri->isOfClass( "URI" ))
         || (i_stream != 0 && ! (i_stream->isNil() || i_stream->isOfClass("Stream")) ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
          .extra( "S|URI,[Stream]" ) );
   }

   Mod::CurlHandle* ca = new Mod::CurlHandle( vm->findWKI("Handle")->asClass() );

   internal_curl_init( vm, ca, i_uri );

   if( i_stream == 0 || i_stream->isNil() )
      ca->setOnDataGetString();
   else
      ca->setOnDataStream(
            dyncast<Stream*>(i_stream->asObject()->getFalconData()) );

  CURLcode retval = curl_easy_perform(ca->handle());
  if( retval != CURLE_OK )
  {
     ca->cleanup();
     ca->gcMark(1); // let the gc kill it
     throw_error( FALCON_ERROR_CURL_EXEC, __LINE__, FAL_STR( curl_err_exec ), retval );
  }

  ca->cleanup();

  if( i_stream == 0 || i_stream->isNil() )
     vm->retval( ca->getData() );

  ca->gcMark(1); // let the gc kill it
}


static void internal_handle_add( VMachine*vm, Item* i_handle )
{
   if( i_handle == 0 || ! i_handle->isOfClass( "Handle" ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                     .extra( "Handle" ) );
   }

   Mod::CurlMultiHandle* mh = dyncast< Mod::CurlMultiHandle* >(
                 vm->self().asObject() );
   Mod::CurlHandle* sh = dyncast< Mod::CurlHandle* >(i_handle->asObjectSafe());

   if( ! mh->addHandle(sh) )
   {
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_HISIN, __LINE__)
            .desc( FAL_STR( curl_err_easy_already_in ) ) );
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

FALCON_FUNC  Multi_init ( ::Falcon::VMachine *vm )
{
   for ( int i = 0; i < vm->paramCount(); ++i )
   {
      internal_handle_add( vm, vm->param(i) );
   }
}


/*#
   @method add Multi
   @brief Adds an @a Handle instance to the multi interface.
   @param h The @a Handle instance to be added.

   Adds a handle to an existing curl multihandle.
*/

FALCON_FUNC  Multi_add ( ::Falcon::VMachine *vm )
{
   Item* i_handle = vm->param(0);
   internal_handle_add( vm, i_handle );
}


/*#
   @method remove Multi
   @brief Adds an @a Handle instance to the multi interface.
   @param h The @a Handle instance to be added.

   Adds a handle to an existing curl multihandle.
*/

FALCON_FUNC  Multi_remove ( ::Falcon::VMachine *vm )
{
   Item* i_handle = vm->param(0);
   if( i_handle == 0 || ! i_handle->isOfClass( "Handle" ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "Handle" ) );
   }

   Mod::CurlMultiHandle* mh = dyncast< Mod::CurlMultiHandle* >(
                   vm->self().asObject() );
   Mod::CurlHandle* sh = dyncast< Mod::CurlHandle* >(i_handle->asObjectSafe());

   if( ! mh->removeHandle(sh) )
   {
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_HNOIN, __LINE__)
            .desc( FAL_STR( curl_err_easy_not_in ) ) );
   }
}


/*#
   @method perform Multi
   @brief Starts or proceeds with the transfers.
   @return The count of remaining operations to be handled.

   The calling application should call repeatedly this method
   until it returns 0, indicating that all the transfers are
   compelete.
*/
FALCON_FUNC  Multi_perform ( ::Falcon::VMachine *vm )
{
   Mod::CurlMultiHandle* mh = dyncast< Mod::CurlMultiHandle* >(
                   vm->self().asObject() );

   int rh = 0;
   CURLMcode ret;
   do{
      ret = curl_multi_perform( mh->handle(), &rh );
   }
   while ( ret == CURLM_CALL_MULTI_PERFORM );

   if ( ret != CURLM_OK )
   {
      throw_merror( FALCON_ERROR_CURL_MULTI, __LINE__, FAL_STR( curl_err_multi_error), ret );
   }
   vm->retval( rh );
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

FALCON_FUNC  CurlError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Mod::CurlError );

   ::Falcon::core::Error_init( vm );
}

}
}

/* end of curl_mod.cpp */
