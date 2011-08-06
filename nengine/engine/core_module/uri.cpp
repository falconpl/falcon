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

/** We keep a C++ uri + the path, the auth data and the query*/
class URICarrier
{
public:
   URI m_uri;
   Path* m_path;
   URI::Authority* m_auth;
   URI::Query* m_query;
   
   uint32 m_gcMark;
   
   URICarrier():
      m_path(0),
      m_auth(0),
      m_query(0)
   {}
   
   URICarrier( const URICarrier& other ):
      m_uri(other.m_uri),
      m_path(0),
      m_auth(0),
      m_query(0)
   {}
   
   ~URICarrier()
   {
      delete m_path;
      delete m_auth;
      delete m_query;
   }
};


ClassURI::ClassURI():
   Class("URI")
{}

ClassURI::~ClassURI()
{}


void ClassURI::dispose( void* self ) const
{
   delete static_cast<URICarrier*>(self);
}


void* ClassURI::clone( void* self ) const
{
   return new URICarrier( *static_cast<URICarrier*>(self) );
}

   
void ClassURI::serialize( DataWriter*, void* ) const
{
   // TODO
}

void* ClassURI::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}
   

void ClassURI::describe( void* instance, String& target, int, int ) const
{
   URICarrier* uc = static_cast<URICarrier*>(instance);
   target = uc->m_uri.encode();
}
   

void ClassURI::gcMark( void* self, uint32 mark ) const
{
   URICarrier* uc = static_cast<URICarrier*>(self);
   uc->m_gcMark = mark;
}

bool ClassURI::gcCheck( void* self, uint32 mark ) const
{
   URICarrier* uc = static_cast<URICarrier*>(self);
   return uc->m_gcMark >= mark;
}


void ClassURI::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   
   URICarrier* uc;
   
   if ( pcount >= 1 )
   {
      Item& other = ctx->opcodeParam(pcount-1);
      if( other.isString() )
      {
         uc = new URICarrier;
         if( ! uc->m_uri.parse( *other.asString() ) )
         {
            delete uc;
            throw new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
               .extra( "invalid URI" )
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
      uc = new URICarrier;
   }
      
   ctx->stackResult( pcount+1, FALCON_GC_STORE(coll, this, uc) );
}


void ClassURI::op_toString( VMContext* ctx, void* self ) const
{
   URICarrier* uc = static_cast<URICarrier*>(self);
   ctx->topData() = (new String(uc->m_uri.encode()))->garbage();
}
 

void ClassURI::op_getProperty( VMContext* ctx, void* self, const String& prop) const
{
   URICarrier* uc = static_cast<URICarrier*>(self);
  
   if( prop == "scheme" )
   {
      ctx->topData() = uc->m_uri.scheme();
   }
   else if( prop == "auth" )
   {
      ctx->topData() = uc->m_uri.auth();
   }
   else if( prop == "path" )
   {
      ctx->topData() = uc->m_uri.path();
   }
   else if( prop == "query" )
   {
      ctx->topData() = uc->m_uri.query();
   }
   else if( prop == "fragment" )
   {
      ctx->topData() = uc->m_uri.fragment();
   }
   else
   {
      return Class::op_getProperty( ctx, self, prop );
   }
}


void ClassURI::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   URICarrier* uc = static_cast<URICarrier*>(self);
  
   if( prop == "scheme" )
   {
      ctx->topData() = uc->m_uri.scheme();
   }
   else if( prop == "auth" )
   {
      ctx->topData() = uc->m_uri.auth();
   }
   else if( prop == "path" )
   {
      ctx->topData() = uc->m_uri.path();
   }
   else if( prop == "query" )
   {
      ctx->topData() = uc->m_uri.query();
   }
   else if( prop == "fragment" )
   {
      ctx->topData() = uc->m_uri.fragment();
   }
   else
   {
      return Class::op_setProperty( ctx, self, prop );
   }
}


}
}

/* end of uri.cpp */
