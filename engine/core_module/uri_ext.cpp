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

#include "core_module.h"
#include <falcon/eng_messages.h>
#include <falcon/crobject.h>

namespace Falcon {
namespace core {

CoreObject* UriObjectFactory( const CoreClass *me, void *uri, bool dyn )
{
   return new UriObject( me, static_cast<URI*>( uri ), dyn );
}

/*# @class URI
   @brief Interface to RFC3986 Universal Resource Indicator.
   @optparam path The URI that will be used as initial data.
   @optparam decode True if the path is URI encoded, and must be decoded (default).
   @raise ParamError in case the inital URI is malformed.

   This class offers an object oriented interface to access
   URI elements.

   Setting the properties in this class immediately reflects on the
   related fields; for example setting the value of the @b uri
   property causes a complete re-parse of the item; setting a field
   as the query string will cause the uri to change.

   Each update is subject to RFC3986 compliance checks, and will raise
   a ParseError if conformance of the URI object is broken.

   @prop scheme URI scheme.
   @prop userInfo User, password or account specification preceding '\@' host.
   @prop host Host specificator.
   @prop port Optional port specificator (following the host after a ':').
   @prop path Path specificator.
   @prop query Query string in the URI.
   @prop fragment Fragment string in the uri (following path and query after a '#').
   @prop uri Complete URI.
*/


UriObject::UriObject( const UriObject &other ):
   CRObject( other )
{
   m_user_data = new URI( *other.getUri() );
   reflectFrom( m_user_data );
}

UriObject::~UriObject()
{
   delete getUri();
}

CoreObject *UriObject::clone() const
{
   return new UriObject( *this );
}


inline void stringize( Item &v, const String &s )
{
   if( ! v.isString() || ( *v.asString() != s ))
   {
      if ( s == "" )
         v.setNil();
      else
         v.setString( new CoreString( s ) );
   }
}

void UriObject::reflectFrom( void *user_data )
{
   const URI &uri = *static_cast<URI*>( user_data );

   stringize( *cachedProperty( "scheme" ), uri.scheme() );
   stringize( *cachedProperty( "userInfo" ), uri.userInfo() );
   stringize( *cachedProperty( "path" ), uri.path() );
   stringize( *cachedProperty( "host" ), uri.host() );
   stringize( *cachedProperty( "port" ), uri.port() );
   stringize( *cachedProperty( "query" ), uri.query() );
   stringize( *cachedProperty( "fragment" ), uri.fragment() );
}


bool UriObject::setProperty( const String &prop, const Item &value )
{
   URI &uri = *getUri();

   if ( prop == "scheme" )
   {
      if ( ! value.isString() ) goto complain;
      uri.scheme( *value.asString() );
   }
   else if ( prop == "userInfo" )
   {
      if ( ! value.isString() ) goto complain;
      uri.userInfo( *value.asString() );
   }
   else if ( prop == "path" )
   {
      if ( ! value.isString() ) goto complain;
      uri.path( *value.asString() );
   }
   else if ( prop == "host" )
   {
      if ( ! value.isString() ) goto complain;
      uri.host( *value.asString() );
   }
   else if ( prop == "port" )
   {
      if ( ! value.isString() ) goto complain;
      uri.port( *value.asString() );
   }
   else if ( prop == "query" )
   {
      if ( ! value.isString() ) goto complain;
      uri.query( *value.asString() );
   }
   else if ( prop == "fragment" )
   {
      if ( ! value.isString() ) goto complain;
      uri.fragment( *value.asString() );
   }
   else {
      // fallback
      return CRObject::setProperty( prop, value );
   }

   if ( ! uri.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new AccessError( ErrorParam( e_param_range, __LINE__).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_uri ) : "" ) );
   }

   return CacheObject::setProperty( prop, value );

   complain:
      throw new AccessError( ErrorParam( e_param_type, __LINE__).
               extra( "S" ) );
}

void UriObject::reflectTo( void* user_data ) const
{
   URI &uri = *static_cast<URI*>( user_data );

   Item *prop = cachedProperty( "scheme" );
   if ( prop->isNil() ) uri.scheme( "" );
   else if ( ! prop->isString() ) goto complain;
   else uri.scheme( *prop->asString() );

   prop = cachedProperty( "userInfo" );
   if ( prop->isNil() ) uri.userInfo( "" );
   else if ( ! prop->isString() ) goto complain;
   else uri.userInfo( *prop->asString() );

   prop = cachedProperty( "path" );
   if ( prop->isNil() ) uri.path( "" );
   else if ( ! prop->isString() ) goto complain;
   else uri.path( *prop->asString() );

   prop = cachedProperty( "host" );
   if ( prop->isNil() ) uri.host( "" );
   else if ( ! prop->isString() ) goto complain;
   else uri.host( *prop->asString() );

   prop = cachedProperty( "port" );
   if ( prop->isNil() ) uri.port( "" );
   else if ( ! prop->isString() ) goto complain;
   else uri.port( *prop->asString() );

   prop = cachedProperty( "query" );
   if ( prop->isNil() ) uri.query( "" );
   else if ( ! prop->isString() ) goto complain;
   else uri.query( *prop->asString() );

   prop = cachedProperty( "fragment" );
   if ( prop->isNil() ) uri.fragment( "" );
   else if ( ! prop->isString() ) goto complain;
   else uri.port( *prop->asString() );

   if ( ! uri.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new CodeError( ErrorParam( e_prop_invalid, __LINE__).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_uri ) : "" ) );
   }

complain:
   throw new CodeError( ErrorParam( e_prop_invalid, __LINE__) );
}


// Reflective URI method
void URI_uri_rfrom(CoreObject *co, void *user_data, Item &property, const PropEntry& )
{
   URI &uri = *static_cast<URI *>( user_data );
   UriObject *instance = dyncast<UriObject*>( co );
   instance->reflectFrom( user_data );

   FALCON_REFLECT_STRING_FROM( (&uri), get );
}


// Reflective URI method
void URI_uri_rto(CoreObject *co, void *user_data, Item &property, const PropEntry& )
{
   URI &uri = *static_cast<URI *>( user_data );
   UriObject *instance = dyncast<UriObject*>( co );

   FALCON_REFLECT_STRING_TO( (&uri), parse )

   instance->reflectFrom( user_data );

   if ( ! uri.isValid() )
   {
      VMachine* vm = VMachine::getCurrent();
      throw new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm != 0 ? vm->moduleString( rtl_invalid_uri ) : "" ) );
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
   UriObject *self = dyncast<UriObject*>(vm->self().asObject());
   Item *p0 = vm->param(0);
   Item *i_parse = vm->param(1);


   // nothing to do
   if ( p0 == 0 )
      return;

   // take the URI generated by the factory (can be empty).
   URI *uri = self->getUri();
   if ( ! p0->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "[S]" ) ) );
   }
   else {
      uri->parse( *p0->asString(), false, (i_parse == 0 || i_parse->isTrue()) );
      if ( ! uri->isValid() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
            origin( e_orig_runtime ).
            extra( vm->moduleString( rtl_invalid_uri ) ) ) );
      }
      else
         self->reflectFrom( uri );

   }
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

   CoreString *str = new CoreString;
   URI::URLEncode( *p0->asString(), *str );
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

   CoreString *str = new CoreString;
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
   UriObject *self = dyncast<UriObject*>( vm->self().asObject() );
   URI &uri = *self->getUri();

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
   CoreDict *dict = new LinearDict( count );
   CoreString *key = new CoreString;
   CoreString *value = new CoreString;
   uri.firstField( *key, *value );
   count--;
   dict->insert( key, value );
   while( count > 0 )
   {
      key = new CoreString;
      value = new CoreString;
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
   UriObject *self = dyncast<UriObject*>( vm->self().asObject() );
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
   self->reflectFrom( &uri );
}

}
}

/* end of uri_ext.cpp */
