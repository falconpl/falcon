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
      String code = FAL_STR( curl_err_desc );
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_INIT, __LINE__ )
            .desc( FAL_STR( curl_err_init ) )
            .extra( code.A(" ").N(retval) ));
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
      String code = FAL_STR( curl_err_desc );
      throw new Mod::CurlError( ErrorParam( FALCON_ERROR_CURL_EXEC, __LINE__ )
            .desc( FAL_STR( curl_err_exec ) )
            .extra( code.A(" ").N(retval) ));
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
