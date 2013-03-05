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
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/function.h>

#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/accesserror.h>

namespace Falcon {
namespace Ext {


//====================================================
// Methods
//
class MethodSetq: public Function
{
public:
   MethodSetq():
      Function( "setq" )
   {
      signature("X,X");
      addParam("key");
      addParam("value");
   }         
   virtual ~MethodSetq() {}
   
   virtual void invoke( VMContext* ctx, int32 pCount = 0 );
};

   
class MethodGetq: public Function
{
public:
   MethodGetq( ):
      Function( "getq" )
   {
      signature("X,[X]");
      addParam("key");
      addParam("default");
   }         
   virtual ~MethodGetq() {}
   
   virtual void invoke( VMContext* ctx, int32 pCount = 0 );
};

 
//=========================================================
// Properties
//

static void set_encoded( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_uri.parse( *value.asString(), 
                     &uric->m_auth, &uric->m_path, &uric->m_query ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
}


static void get_encoded( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->m_uri.encode() ) ); 
}


static void set_scheme( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);
   uric->m_uri.scheme(*value.asString());
}


static void get_scheme( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->m_uri.scheme() ) ); 
}


static void set_auth( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_auth.parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
   
   uric->m_uri.auth( uric->m_auth.encode() );
}


static void get_auth( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);   
   value = FALCON_GC_HANDLE( new String( uric->m_uri.auth() ));
}



static void set_path( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_path.parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
   
   uric->m_uri.path( uric->m_path.encode() );
}


static void get_path( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);      
   value = FALCON_GC_HANDLE( new String( uric->m_uri.path() ));
}


static void set_query( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);
   if( ! uric->m_query.parse( *value.asString() ) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }   
   
   uric->m_uri.query( uric->m_query.encode() );
}


static void get_query( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->m_uri.query() ));
}


static void set_host( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);   
      
   uric->m_auth.host( *value.asString() );   
   uric->m_uri.auth( uric->m_auth.encode() );
}

static void get_host( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->m_auth.host() ));
}


static void set_fragment( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);      
   uric->m_uri.fragment( *value.asString() );
}


static void get_fragment( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->m_uri.fragment() ));
}



static void set_port( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);
   uric->m_auth.port( *value.asString() );
   uric->m_uri.auth( uric->m_auth.encode() );
}


static void get_port( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);   
   value = FALCON_GC_HANDLE( new String( uric->m_auth.port() ));
}

   
static void set_user( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);   
   uric->m_auth.user( *value.asString() );
   uric->m_uri.auth( uric->m_auth.encode() );
}


static void get_user( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->m_auth.user() ));
}


static void set_pwd( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   URICarrier* uric = static_cast<URICarrier*>(instance);
   uric->m_auth.password( *value.asString() );   
   uric->m_uri.auth( uric->m_auth.encode() );
}


static void get_pwd( const Class*, const String&, void* instance, Item& value )
{
   URICarrier* uric = static_cast<URICarrier*>(instance);
   value = FALCON_GC_HANDLE( new String( uric->m_auth.password() ));
}   
 

//==================================================================
// Main class
//==================================================================

ClassURI::ClassURI():
   Class("URI")
{   
   addProperty( "encoded", &get_encoded, &set_encoded );
   addProperty( "scheme", &get_scheme, &set_scheme );
   addProperty( "auth", &get_auth, &set_auth );
   addProperty( "path", &get_path, &set_path );
   addProperty( "query", &get_query, &set_query );
   addProperty( "fragment", &get_fragment, &set_fragment );
   addProperty( "host", &get_host, &set_host );
   addProperty( "port", &get_port, &set_port );
   addProperty( "user", &get_user, &set_user );
   addProperty( "pwd", &get_pwd, &set_pwd );  
   
   addMethod( new MethodGetq );
   addMethod( new MethodSetq );
}

ClassURI::~ClassURI()
{}

 
void ClassURI::dispose( void* instance ) const
{
   URICarrier* uc = static_cast<URICarrier*>(instance);
   delete uc;
}

void* ClassURI::clone( void* instance ) const
{
   URICarrier* uc = static_cast<URICarrier*>(instance);
   return new URICarrier(*uc);
}

void ClassURI::gcMarkInstance( void* instance, uint32 mark ) const
{
   URICarrier* uc = static_cast<URICarrier*>(instance);
   uc->m_mark = mark;
}

bool ClassURI::gcCheckInstance( void* instance, uint32 mark ) const
{
   URICarrier* uc = static_cast<URICarrier*>(instance);
   return uc->m_mark >= mark;
}


void ClassURI::store( VMContext*, DataWriter* stream, void* instance ) const
{
   URICarrier* uc = static_cast<URICarrier*>(instance);
   stream->write(uc->m_uri.encode());
}


void ClassURI::restore( VMContext* ctx, DataReader* stream ) const
{
   String uriName;
   stream->read( uriName );
   URICarrier* uc = new URICarrier();
   try {
      uc->m_uri.parse( uriName );
      ctx->pushData( Item( this, uc ) );
   }
   catch( ... ) {
      delete uc;
      throw;
   }
}
   
void* ClassURI::createInstance() const
{
   return new URICarrier();
}


bool ClassURI::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   URICarrier* uc = static_cast<URICarrier*>(instance );
   
   if ( pcount >= 1 )
   {
      Item& other = *ctx->opcodeParams(pcount);
      if( other.isString() )
      {
         if( ! uc->m_uri.parse( *other.asString(), &uc->m_auth, &uc->m_path, &uc->m_query ) )
         {
            delete uc;
            throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC )
               .origin(ErrorParam::e_orig_mod) );
         }
      }
      else if( other.asClass() == this )
      {
         uc->m_uri = static_cast<URICarrier*>(other.asInst())->m_uri;
      }
      else
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
               .extra( "S|URI" )
               .origin(ErrorParam::e_orig_mod) );
      }
   }
   
   return false;
}

void ClassURI::op_toString( VMContext* ctx, void* self ) const
{
   URICarrier* uc = static_cast<URICarrier*>(self);
   ctx->topData() = FALCON_GC_HANDLE(new String(uc->m_uri.encode()));
}

//=========================================================
// Methods
//

void MethodSetq::invoke( VMContext* ctx, int32 )
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
   
      
void MethodGetq::invoke( VMContext* ctx, int32 )  
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
  
}
}

/* end of uri.cpp */
