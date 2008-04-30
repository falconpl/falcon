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

class URICarrier: public UserData
{
public:
   URI m_uri;

   URICarrier() {}

   URICarrier( const URICarrier &other ):
      m_uri( other.m_uri )
      {}

   virtual void getProperty( VMachine *vm, const String &propName, Item &prop );
   virtual void setProperty( VMachine *vm, const String &propName, Item &prop );
   virtual bool isReflective() const;
   virtual UserData *clone() const;
};

bool URICarrier::isReflective() const
{
   return true;
}

UserData *URICarrier::clone() const
{
   return new URICarrier( *this );
}

void URICarrier::setProperty( VMachine *vm, const String &propName, Item &prop )
{
   if( ! prop.isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).extra( "S" ) ) );
      return;
   }

   if ( propName == "scheme" )
   {
      m_uri.scheme( *prop.asString() );
   }
   else if ( propName == "userInfo" )
   {
      m_uri.userInfo( *prop.asString() );
   }
   else if ( propName == "host" )
   {
      m_uri.host( *prop.asString() );
   }
   else if ( propName == "port" )
   {
      m_uri.port( *prop.asString() );
   }
   else if ( propName == "path" )
   {
      m_uri.path( *prop.asString() );
   }
   else if ( propName == "query" )
   {
      m_uri.query( *prop.asString() );
   }
   else if ( propName == "fragment" )
   {
      m_uri.fragment( *prop.asString() );
   }
   else if ( propName == "uri" )
   {
      m_uri.parse( *prop.asString(), false, true );
   }


   if ( ! m_uri.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_invalid_uri ) ) ) );
   }
}

void URICarrier::getProperty( VMachine *vm, const String &propName, Item &prop )
{
   if ( ! m_uri.isValid() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params ).
         origin( e_orig_runtime ).
         extra( vm->moduleString( msg::rtl_invalid_uri ) ) ) );
   }

   if ( propName == "scheme" )
   {
      prop = const_cast<String *>(&m_uri.scheme());
   }
   else if ( propName == "userInfo" )
   {
      prop = const_cast<String *>( &m_uri.userInfo() );
   }
   else if ( propName == "host" )
   {
      prop = const_cast<String *>( &m_uri.host() );
   }
   else if ( propName == "port" )
   {
     prop = const_cast<String *>( &m_uri.port() );
   }
   else if ( propName == "path" )
   {
      prop = const_cast<String *>( &m_uri.path() );
   }
   else if ( propName == "query" )
   {
      if ( m_uri.fieldCount() != 0 )
         m_uri.makeQuery();

      prop = const_cast<String *>( &m_uri.query() );
   }
   else if ( propName == "fragment" )
   {
      prop = const_cast<String *>( &m_uri.fragment() );
   }
   else if ( propName == "uri" )
   {
      prop = const_cast<String *>( &m_uri.get( true ) );
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
   Item *p0 = vm->param(0);

   if ( ( p0 == 0 ) || ( ! p0->isString() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "S" ) ) );
      return;
   }

   // we need anyhow a carrier.
   URICarrier *carrier = new URICarrier;
   CoreObject *self = vm->self().asObject();
   self->setUserData( carrier );

   carrier->m_uri.parse( *p0->asString() );
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
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_invalid_uri ) ) ) );
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
   URICarrier *carrier = (URICarrier *) self->getUserData();
   URI &uri = carrier->m_uri;

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
            origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_invalid_uri ) ) ) );
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
   URICarrier *carrier = (URICarrier *) self->getUserData();
   URI &uri = carrier->m_uri;

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
