/*
   FALCON - The Falcon Programming Language
   FILE: uri_ext.cpp

   Falcon class reflecting URI class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Feb 2008 22:39:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon class reflecting URI class.
*/

#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/uri.h>
#include <falcon/lineardict.h>

#include "falcon_rtl_ext.h"
#include "rtl_messages.h"

/*#
   @beginmodule falcon_rtl
*/

namespace Falcon {
namespace Ext {

/*# @class URI
   @brief Interface to RFC3986 Universal Resource Indicator.
   @optparam path The URI that will be used as initial data.
   @raise ParamError in case the inital URI is malformed.

   This class offers an object oriented interface to access
   URI elements.

   Setting the properties in this class immediately reflects on the
   related fields; in example setting the value of the @b uri
   property causes a complete re-parse of the item; setting a field
   as the query string will cause the uri to change.

   Each update is subject to RFC3986 compliance checks, and will raise
   a ParseError if conformance of the URI object is broken.

   @prop scheme URI scheme.
   @prop userInfo User, password or account specification preceding '@' host.
   @prop host Host specificator.
   @prop port Optional port specificator (following the host after a ':').
   @prop path Path specificator.
   @prop query Query string in the URI.
   @prop fragment Fragment string in the uri (following path and query after a '#').
   @prop uri Complete URI.
*/

void* URIManager::onInit( VMachine *vm )
{
   return 0;
}

void  URIManager::onDestroy( VMachine *vm, void *user_data )
{
   delete static_cast<URI* >( user_data );
}

void* URIManager::onClone( VMachine *vm, void *user_data )
{
   return new URI( *static_cast<URI* >( user_data ) );
}

bool URIManager::onObjectReflectTo( CoreObject *reflector, void *user_data )
{
   URI &uri = *static_cast<URI *>( user_data );

   Item *property = reflector->cachedProperty( "scheme" );
   if ( ! property->isString() )
      goto complain;

   uri.scheme( *property->asString() );

   property = reflector->cachedProperty( "path" );
   if ( ! property->isString() )
      goto complain;

   uri.path( *property->asString() );

   property = reflector->cachedProperty( "userInfo" );
   if ( ! property->isString() )
      goto complain;

   uri.userInfo( *property->asString() );

   property = reflector->cachedProperty( "host" );
   if ( ! property->isString() )
      goto complain;

   uri.host( *property->asString() );

   property = reflector->cachedProperty( "port" );
   if ( ! property->isString() )
      goto complain;

   uri.port( *property->asString() );

   property = reflector->cachedProperty( "query" );
   if ( ! property->isString() )
      goto complain;

   uri.query( *property->asString() );

   property = reflector->cachedProperty( "fragment" );
   if ( ! property->isString() )
      goto complain;

   uri.fragment( *property->asString() );

   if ( ! uri.isValid() )
   {
      reflector->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( reflector->origin()->moduleString( rtl_invalid_uri ) ) ) );
   }

   return true;

complain:
   reflector->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
      origin( e_orig_runtime ).extra( "S" ) ) );
   return true;
}

bool URIManager::onObjectReflectFrom( CoreObject *reflector, void *user_data )
{
   URI &uri = *static_cast<URI *>( user_data );

   reflector->cacheStringProperty( "scheme", uri.scheme() );
   reflector->cacheStringProperty( "userInfo", uri.userInfo() );
   reflector->cacheStringProperty( "path", uri.path() );
   reflector->cacheStringProperty( "host", uri.host() );
   reflector->cacheStringProperty( "port", uri.port() );
   reflector->cacheStringProperty( "query", uri.query() );
   reflector->cacheStringProperty( "fragment", uri.fragment() );

   // TODO: reflect URI
   return true;
}

// Reflective URI method
void URI_uri_rfrom(CoreObject *instance, void *user_data, Item &property )
{
   URI &uri = *static_cast<URI *>( user_data );
   instance->reflectTo( user_data );

   FALCON_REFLECT_STRING_FROM( (&uri), get )
}


// Reflective URI method
void URI_uri_rto(CoreObject *instance, void *user_data, Item &property )
{
   URI &uri = *static_cast<URI *>( user_data );

   FALCON_REFLECT_STRING_TO( (&uri), parse )

   instance->reflectFrom( user_data );

   if ( ! uri.isValid() )
   {
      instance->origin()->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( instance->origin()->moduleString( rtl_invalid_uri ) ) ) );
   }
}

/*# @init URI
   @brief Constructor for the URI class.
   @raise ParamError in case the inital URI is malformed.

   Builds the URI object, optionally using the given parameter
   as a complete URI constructor.
*/
FALCON_FUNC  URI_init ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *p0 = vm->param(0);
   URI *uri;

   if ( ( p0 == 0 ) || ( ! p0->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S" ) ) );
      uri = new URI;
   }
   else {
      uri = new URI( *p0->asString() );
      if ( ! uri->isValid() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( vm->moduleString( rtl_invalid_uri ) ) ) );
      }
      else
         self->reflectFrom( uri );

   }

   self->setUserData( uri );
}



/*# @method encode URI
   @brief Encode a string to URL encoding (static).
   @param string The string to be encoded.
   @return the URL/URI encoded string.
*/

FALCON_FUNC  URI_encode ( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param(0);

   if ( ( p0 == 0 ) || ( ! p0->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S" ) ) );
      return;
   }

   String *str = new GarbageString( vm );
   URI::URLDecode( *p0->asString(), *str );
   vm->retval( str );
}

/*# @method decode URI
   @brief Decode a string to from URL encoding (static).
   @param enc_string The URI/URL encoded string.
   @return The decoded string.
   @raise ParamError if the string is not a valid URI/URL encoded string.
*/
FALCON_FUNC  URI_decode ( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param(0);

   if ( ( p0 == 0 ) || ( ! p0->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S" ) ) );
      return;
   }

   String *str = new GarbageString( vm );
   if ( ! URI::URLDecode( *p0->asString(), *str ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_invalid_uri ) ) ) );
      return;
   }

   vm->retval( str );
}


/*# @method getFields URI
   @brief Returns fields contained in the query element into a dictionary.
   @return The fields as a dictionary of nil if the query part contains no element.
   @raise ParamError if the string is not a valid URI/URL encoded string.
*/
FALCON_FUNC  URI_getFields ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   URI &uri = * (URI *) self->getUserData();

   if ( uri.query().size() == 0 )
   {
      vm->retnil();
      return;
   }

   if( uri.fieldCount() == 0 )
   {
      // we have a query but no fields; this means we still have to parse it.
      if ( ! uri.parseQuery( true ) )
      {
         // todo: better signalation
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            origin( e_orig_runtime ).extra( vm->moduleString( rtl_invalid_uri ) ) ) );
         return;
      }

      // really nothing to parse?
      if ( uri.fieldCount() == 0 )
      {
         vm->retnil();
         return;
      }
   }

   // ok, build our dictionary
   uint32 count = uri.fieldCount();
   CoreDict *dict = new LinearDict( vm, count );
   String *key = new GarbageString( vm );
   String *value = new GarbageString( vm );
   uri.firstField( *key, *value );
   count--;
   dict->insert( key, value );
   while( count > 0 )
   {
      key = new GarbageString( vm );
      value = new GarbageString( vm );
      uri.nextField( *key, *value );
      count --;
      dict->insert( key, value );
   }

   vm->retval( dict );
}

/*# @method setFields URI
   @brief Sets query fields for this uri.
   @param fields A dictionary of fields or nil to clear the query.
   @raise ParamError if the input dictionary contains non-string values.
*/
FALCON_FUNC  URI_setFields ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   URI &uri = *(URI *) self->getUserData();

   Item *p0 = vm->param(0);

   if ( ( p0 == 0 ) || ( ! p0->isDict() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S" ) ) );
      return;
   }

   CoreDict *dict = p0->asDict();
   dict->traverseBegin();
   Item key, value;

   while( dict->traverseNext( key, value ) )
   {
      if ( ( !key.isString()) || (! value.isString() ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S" ) ) );
         return;
      }

      uri.setField( *key.asString(), *value.asString() );
   }

   uri.makeQuery();
}

}
}

/* end of uri_ext.cpp */
