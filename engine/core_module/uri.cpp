/*
   FALCON - The Falcon Programming Language.
   FILE: uri.cpp

   Falcon core module -- Interface to URI class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/uri.cpp"

#include <falcon/cm/uri.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/codeerror.h>
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/paramerror.h>

namespace Falcon {
namespace Ext {


ClassURI::ClassURI():
   ClassUser("URI"),
   m_mthSetq( this ),
   m_mthGetq( this ),
   m_encoded( this ),
   m_scheme( this ),
   m_auth( this ),
   m_path( this ),
   m_query( this ),
   m_fragment( this ),
   m_propHost( this ),
   m_propPort( this ),
   m_propUser( this ),
   m_propPwd( this )
{
}

ClassURI::~ClassURI()
{}

   
void ClassURI::serialize( DataWriter*, void* ) const
{
   // TODO
}

void* ClassURI::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}
   
void* ClassURI::createInstance( Item* params, int pcount ) const
{
   URICarrier* uc;
   
   if ( pcount >= 1 )
   {
      Item& other = *params;
      if( other.isString() )
      {
         uc = new URICarrier( carriedProps() );         
         if( ! uc->m_uri.parse( *other.asString(), &uc->m_auth, &uc->m_path, &uc->m_query ) )
         {
            delete uc;
            throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC )
               .origin(ErrorParam::e_orig_mod) );
         }
      }
      else if( other.asClass() == this )
      {
         uc = new URICarrier( *static_cast<URICarrier*>(other.asInst()) );
      }
      else
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
               .extra( "S|URI" )
               .origin(ErrorParam::e_orig_mod) );
      }
      
   }
   else
   {
      uc = new URICarrier( carriedProps() );
   }
      
   return uc;
}


void ClassURI::op_toString( VMContext* ctx, void* self ) const
{
   URICarrier* uc = static_cast<URICarrier*>(self);
   ctx->topData() = (new String(uc->m_uri.encode()))->garbage();
}

//=========================================================
// Methods
//

void ClassURI::MethodSetq::invoke( VMContext* ctx, int32 )
{
   Item* iKey = ctx->param(0);
   Item* iValue = ctx->param(1);
   
   if( iKey == 0 || ! iKey->isString() || 
       iValue == 0 || ! ( iValue->isString() || iValue->isNil() ) )
   {
      throw paramError( __LINE__, SRC );
   }
   
   URICarrier* uc = static_cast<URICarrier*>(ctx->self().asInst());
   if( iValue->isString() )
   {
      uc->m_query.put( *iKey->asString(), *iValue->asString() );
   }
   else
   {
      uc->m_query.remove( *iKey->asString() );
   }
   uc->m_uri.query( uc->m_query.encode() );
   
   ctx->returnFrame();
}
   
      
void ClassURI::MethodGetq::invoke( VMContext* ctx, int32 )  
{
   Item* iKey = ctx->param(0);
   Item* iValue = ctx->param(1);
   
   if( iKey == 0 || ! iKey->isString() )
   {
      throw paramError( __LINE__, SRC );
   }
   
   URICarrier* uc = static_cast<URICarrier*>(ctx->self().asInst());
   
   String value;
   if( uc->m_query.get( *iKey->asString(), value ) )
   {
      ctx->returnFrame( value );
   }
   else
   {
      if( iValue != 0 )
      {
         ctx->returnFrame( *iValue );
      }
      else
      {
         ctx->returnFrame();
      }
   }
}

//=========================================================
// Properties
//

void ClassURI::PropertyEncoded::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_uri.parse( *value.asString(), 
                     &uric->m_auth, &uric->m_path, &uric->m_query ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
}


const String& ClassURI::PropertyEncoded::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   return uric->m_uri.encode();      
}


void ClassURI::PropertyAuth::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_auth.parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
   
   uric->m_uri.auth( uric->m_auth.encode() );
}


const String& ClassURI::PropertyAuth::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);   
   return uric->m_uri.auth();
}



void ClassURI::PropertyPath::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_path.parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
   
   uric->m_uri.path( uric->m_path.encode() );
}


const String& ClassURI::PropertyPath::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);      
   return uric->m_uri.path();
}


void ClassURI::PropertyQuery::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_query.parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
   
   uric->m_uri.query( uric->m_query.encode() );
}


const String& ClassURI::PropertyQuery::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   return uric->m_uri.query();
}


void ClassURI::PropertyHost::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);   
      
   uric->m_auth.host( *value.asString() );   
   uric->m_uri.auth( uric->m_auth.encode() );
}

const String& ClassURI::PropertyHost::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   return uric->m_auth.host();
}

void ClassURI::PropertyPort::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);
   uric->m_auth.port( *value.asString() );
   uric->m_uri.auth( uric->m_auth.encode() );
}

const String& ClassURI::PropertyPort::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);   
   return uric->m_auth.port();
}

   
void ClassURI::PropertyUser::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);   
   uric->m_auth.user( *value.asString() );
   uric->m_uri.auth( uric->m_auth.encode() );
}

const String& ClassURI::PropertyUser::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   return uric->m_auth.user();
}


void ClassURI::PropertyPwd::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   URICarrier* uric = static_cast<URICarrier*>(instance);
   uric->m_auth.password( *value.asString() );   
   uric->m_uri.auth( uric->m_auth.encode() );
}

const String& ClassURI::PropertyPwd::getString( void* instance )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   return uric->m_auth.password();
}   
   
}
}

/* end of uri.cpp */
