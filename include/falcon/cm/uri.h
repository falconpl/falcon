/*
   FALCON - The Falcon Programming Language.
   FILE: uri.h

   Falcon core module -- Interface to URI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_URI_H
#define FALCON_CORE_URI_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/classes/classuser.h>

#include <falcon/usercarrier.h>
#include <falcon/uri.h>
#include <falcon/path.h>

namespace Falcon {
namespace Ext {

/** We keep a C++ uri + the path, the auth data and the query*/
class FALCON_DYN_CLASS URICarrier: public UserCarrier
{
public:
   URI m_uri;
   Path m_path;
   URI::Authority m_auth;
   URI::Query m_query;
   
   URICarrier( uint32 nprops ):
      UserCarrier(nprops)
   {}
   
   URICarrier( const URICarrier& other ):
      UserCarrier( other.dataSize() ),
      m_uri( other.m_uri )
   {}
   
   virtual ~URICarrier()
   {      
   }
   
   virtual URICarrier* clone() const { return new URICarrier(*this); }
};


class FALCON_DYN_CLASS ClassURI: public ClassUser
{
public:
   
   ClassURI();
   virtual ~ClassURI();

   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;
   
   //=============================================================
   //
   virtual void* createInstance( Item* params, int pcount ) const;   
   virtual void op_toString( VMContext* ctx, void* self ) const;
   
private:
   //====================================================
   // Methods
   //
   class FALCON_DYN_CLASS MethodSetq: public Method
   {
   public:
      MethodSetq( ClassUser* owner ):
         Method( owner, "setq" )
      {
         signature("X,X");
         addParam("key");
         addParam("value");
      }         
      virtual ~MethodSetq() {}
      
      virtual void invoke( VMContext* ctx, int32 pCount = 0 );
   }
   m_mthSetq;
   
      
   class FALCON_DYN_CLASS MethodGetq: public Method
   {
   public:
      MethodGetq( ClassUser* owner ):
         Method( owner, "getq" )
      {
         signature("X,[X]");
         addParam("key");
         addParam("default");
      }         
      virtual ~MethodGetq() {}
      
      virtual void invoke( VMContext* ctx, int32 pCount = 0 );
   }
   m_mthGetq;
   
   //====================================================
   // Properties.
   //
   
   class FALCON_DYN_CLASS PropertyEncoded: public PropertyString
   {
   public:
      PropertyEncoded( ClassUser* owner ):
         PropertyString( owner, "encoded" )
      {}      
      virtual ~PropertyEncoded() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_encoded;
   
   class FALCON_DYN_CLASS PropertyScheme: public PropertyString
   {
   public:
      PropertyScheme( ClassUser* owner ):
         PropertyString( owner, "scheme" )
      {}
      
      virtual ~PropertyScheme() {}
      
      virtual void set( void* instance, const Item& value )
      {
         checkType( value.isString(), "S" );
         static_cast<URICarrier*>(instance)->m_uri.scheme();
      }
      
      virtual const String& getString( void* instance )
      {
         return static_cast<URICarrier*>(instance)->m_uri.scheme();
      }
   }
   m_scheme;
   
   class FALCON_DYN_CLASS PropertyAuth: public PropertyString
   {
   public:
      PropertyAuth( ClassUser* owner ):
         PropertyString( owner, "auth" )
      {}
      
      virtual ~PropertyAuth() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_auth;
   
   class FALCON_DYN_CLASS PropertyPath: public PropertyString
   {
   public:
      PropertyPath( ClassUser* owner ):
         PropertyString( owner, "path" )
      {}
      
      virtual ~PropertyPath() {}
      
      virtual void set( void* instance, const Item& value );     
      virtual const String& getString( void* instance );
   }
   m_path;
   
   class FALCON_DYN_CLASS PropertyQuery: public PropertyString
   {
   public:
      PropertyQuery( ClassUser* owner ):
         PropertyString( owner, "query" )
      {}
      
      virtual ~PropertyQuery() {}
      
      virtual void set( void* instance, const Item& value );
      virtual const String& getString( void* instance );
   }
   m_query;
   
   class FALCON_DYN_CLASS PropertyFragment: public PropertyString
   {
   public:
      PropertyFragment( ClassUser* owner ):
         PropertyString( owner, "fragment" )
      {}
      
      virtual ~PropertyFragment() {}
      
      virtual void set( void* instance, const Item& value )
      {
         checkType( value.isString(), "S" );
         static_cast<URICarrier*>(instance)->m_uri.fragment();
      }
      
      virtual const String& getString( void* instance )
      {
         return static_cast<URICarrier*>(instance)->m_uri.fragment();
      }
   }
   m_fragment;
   
   class FALCON_DYN_CLASS PropertyHost: public PropertyString
   {
   public:
      PropertyHost( ClassUser* owner ):
         PropertyString( owner, "host" )
      {}      
      virtual ~PropertyHost() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propHost;
   
   class PropertyPort: public PropertyString
   {
   public:
      PropertyPort( ClassUser* owner ):
         PropertyString( owner, "port" )
      {}      
      virtual ~PropertyPort() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propPort;
   
   class FALCON_DYN_CLASS PropertyUser: public PropertyString
   {
   public:
      PropertyUser( ClassUser* owner ):
         PropertyString( owner, "user" )
      {}      
      virtual ~PropertyUser() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propUser;
   
   class FALCON_DYN_CLASS PropertyPwd: public PropertyString
   {
   public:
      PropertyPwd( ClassUser* owner ):
         PropertyString( owner, "pwd" )
      {}      
      virtual ~PropertyPwd() {}
      
      virtual void set( void* instance, const Item& value );      
      virtual const String& getString( void* instance );
   }
   m_propPwd;
   
};

}
}

#endif	/* FALCON_CORE_TOSTRING_H */

/* end of uri.h */
