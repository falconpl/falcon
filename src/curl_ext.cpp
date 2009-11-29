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

#include <falcon/engine.h>
#include <curl/curl.h>

#include "curl_mod.h"
#include "curl_ext.h"
#include "curl_st.h"

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

// The following is a faldoc block for the function
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
*/

FALCON_FUNC  Handle_init( ::Falcon::VMachine *vm )
{
   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );
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

   Item* i_uri = vm->param(0);

   // no parameter? -- nothing to do
   if( i_uri == 0 )
      return;

   CURLcode retval;

   if( i_uri->isString() )
   {
      String enc = URI::URLEncode( *i_uri->asString() );
      AutoCString curi( enc );

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
   @method exec Handle
   @brief Transfers data from the remote.

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
}

/*#
   @method setOutConsole Handle
   @brief Asks for subsequent transfer(s) to be sent to process console (raw stdout).

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
   @method setOutConsole Handle
   @brief Asks for subsequent transfer(s) to be stored in a temporary string.

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
}


/*#
   @method setOutStream Handle
   @brief Asks for subsequent transfer(s) to be stored in a given stream.
   @param stream The stream to be used.

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
}

/*#
   @method setOutCallback Handle
   @brief Asks for subsequent transfer(s) to be handled to a given callback.
   @param cb A callback item that will receive incoming data as a binary string.

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
}

// not yet active
/*
   @method setOutMessage Handle
   @brief Asks for subsequent transfer(s) to be handled as a message broadcast.
   @param msg A string representing a message or a VMSlot.

   This method instructs this handle to perform message broadcast when data
   is received.

   When called, @a Handle.exec will repeatedly broadcast @msg sending two parameters:
   itself (this Handle object) and the received data, as a binary string.

   The string is not encoded in any format, and could be considered filled with binary
   data.
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
}


/*#
   @method setInStream Handle
   @brief Asks for subsequent upload(s) to read data from the given stream.
   @param stream The stream to be used.

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
   case CURLOPT_SOCKS5_GSSAPI_NEC:
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
   case CURLOPT_CERTINFO:
   case CURLOPT_SSL_VERIFYHOST:
   case CURLOPT_SSL_SESSIONID_CACHE:
      {
         long bVal = i_data->isTrue() ? 1 : 0;
         ret = curl_easy_setopt( curl, iOpt, bVal );
      }
      break;


    case CURLOPT_PROTOCOLS:
    case CURLOPT_REDIR_PROTOCOLS:
    case CURLOPT_PROXYPORT:
    case CURLOPT_PROXYTYPE:
    case CURLOPT_LOCALPORT:
    case CURLOPT_LOCALPORTRANGE:
    case CURLOPT_DNS_CACHE_TIMEOUT:
    case CURLOPT_DNS_USE_GLOBAL_CACHE:
    case CURLOPT_BUFFERSIZE:
    case CURLOPT_PORT:
    case CURLOPT_ADDRESS_SCOPE:
    case CURLOPT_NETRC:
    case CURLOPT_HTTPAUTH:
    case CURLOPT_PROXYAUTH:
    case CURLOPT_MAXREDIRS:
    case CURLOPT_POSTREDIR:
    case CURLOPT_HTTP_VERSION:
    case CURLOPT_HTTP_CONTENT_DECODING:
    case CURLOPT_HTTP_TRANSFER_DECODING:
    case CURLOPT_TFTP_BLKSIZE:
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
    case CURLOPT_NOPROXY:
    case CURLOPT_SOCKS5_GSSAPI_SERVICE:
    case CURLOPT_INTERFACE:
    case CURLOPT_NETRC_FILE:
    case CURLOPT_USERPWD:
    case CURLOPT_PROXYUSERPWD:
    case CURLOPT_USERNAME:
    case CURLOPT_PASSWORD:

    case CURLOPT_PROXYUSERNAME:
    case CURLOPT_PROXYPASSWORD:
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
    case CURLOPT_ISSUERCERT:
    case CURLOPT_CAPATH:
    case CURLOPT_CRLFILE:
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
            .extra( "I;X" ) );
   }

   // setup our options
   Mod::CurlHandle* h = dyncast< Mod::CurlHandle* >( vm->self().asObject() );

   if ( h->handle() == 0 )
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_PM, __LINE__ )
            .desc( FAL_STR( curl_err_pm ) ) );

   CURLoption iOpt = (CURLoption) i_option->asInteger();
   internal_setOpt( vm, h, iOpt, i_data );
}


/*#
   @method setOptions Handle
   @brief Sets a list of cURL option for this specific handle.
   @param opts A dictionary of options, where each key is an option number, and its value is the option value.
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

   \note The data will be sent not encoded in any particular format (it will be
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
   @class CurlError
   @brief Error generated by cURL while operating.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   See the Error class in the core module.
*/

/*#
   @init CurlError
   @brief Initializes the process error.
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
