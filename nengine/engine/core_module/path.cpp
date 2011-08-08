/*
   FALCON - The Falcon Programming Language.
   FILE: path.cpp

   Falcon core module -- Interface to Path class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/path.cpp"

#include <falcon/cm/path.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/codeerror.h>
#include <falcon/path.h>
#include <falcon/paramerror.h>

#include "falcon/cm/uri.h"

namespace Falcon {
namespace Ext {


ClassPath::ClassPath():
   ClassUser("Path"),
   m_propRes(this),
   m_propLoc(this),
   m_propFullLoc(this),
   m_propFile(this),
   m_propExt(this),
   m_propFileExt(this),
   m_propEncoded(this),
   
   m_propWLoc(this),
   m_propFileWFullLoc(this),   
   m_propFileWEncoded(this),
   
   FALCON_INIT_METHOD( absolutize ),
   FALCON_INIT_METHOD( relativize ),
   FALCON_INIT_METHOD( canonicize ),
   FALCON_INIT_METHOD( cwd )
{
}

ClassPath::~ClassPath()
{}

   
void ClassPath::serialize( DataWriter*, void* ) const
{
   // TODO
}

void* ClassPath::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}
   
void* ClassPath::createInstance( Item* params, int pcount ) const
{
   PathCarrier* uc;
   
   if ( pcount >= 1 )
   {
      Item& other = *params;
      if( other.isString() )
      {
         uc = new PathCarrier( carriedProps() );         
         if( ! uc->m_path.parse( *other.asString() ) )
         {
            delete uc;
            throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC )
               .origin(ErrorParam::e_orig_mod) );
         }
      }
      else if( other.asClass() == this )
      {
         uc = new PathCarrier( *static_cast<PathCarrier*>(other.asInst()) );
      }
      else
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
               .extra( "S|Path" )
               .origin(ErrorParam::e_orig_mod) );
      }
      
   }
   else
   {
      uc = new PathCarrier( carriedProps() );
   }
      
   return uc;
}


void ClassPath::op_toString( VMContext* ctx, void* self ) const
{
   PathCarrier* uc = static_cast<PathCarrier*>(self);
   ctx->topData().setString( uc->m_path.encode() ); // garbages
}


//====================================================
// Properties.
//
   
void ClassPath::PropertyResource::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.resource(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyResource::getString( void* instance )
{
   return static_cast<PathCarrier*>(instance)->m_path.resource();
}


void ClassPath::PropertyLocation::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.location(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyLocation::getString( void* instance )
{
   return static_cast<PathCarrier*>(instance)->m_path.location();
}


void ClassPath::PropertyFullLocation::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.fulloc(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyFullLocation::getString( void* instance )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_path.getFullLocation(pc->m_fulloc);   
   return pc->m_fulloc;
}
   
  
void ClassPath::PropertyFile::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.file(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyFile::getString( void* instance )
{
   return static_cast<PathCarrier*>(instance)->m_path.file();
}
   

void ClassPath::PropertyExt::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.ext(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyExt::getString( void* instance )
{
   return static_cast<PathCarrier*>(instance)->m_path.ext();
}

void ClassPath::PropertyFileExt::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.fileext(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyFileExt::getString( void* instance )
{
   return static_cast<PathCarrier*>(instance)->m_path.fileext();
}


void ClassPath::PropertyEncoded::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.parse(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyEncoded::getString( void* instance )
{
   return static_cast<PathCarrier*>(instance)->m_path.encode();
}


void ClassPath::PropertyWLoc::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.location(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyWLoc::getString( void* instance )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_path.getWinLocation(pc->m_winLoc);   
   return pc->m_winLoc;
}

void ClassPath::PropertyWFullLoc::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.fulloc(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyWFullLoc::getString( void* instance )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_path.getFullWinLocation(pc->m_fullWinLoc);   
   return pc->m_fullWinLoc;
}


void ClassPath::PropertyWEncoded::set( void* instance, const Item& value )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.parse(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

const String& ClassPath::PropertyWEncoded::getString( void* instance )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_path.getWinFormat(pc->m_winpath);   
   return pc->m_winpath;
}
   

FALCON_DEFINE_METHOD_P1( ClassPath, absolutize )
{
   Item* i_path = ctx->param(0);
   if( i_path != 0 && !(i_path->isString()||i_path->isNil()) )
   {
      throw paramError();
   }
   
   PathCarrier* pc = static_cast<PathCarrier*>(ctx->self().asInst());
   if( i_path == 0 || i_path->isNil() )
   {
      pc->m_path.absolutize();
   }
   else
   {
      pc->m_path.absolutize( *i_path->asString() );
   }
   
   ctx->returnFrame();   
}


FALCON_DEFINE_METHOD_P1( ClassPath, relativize )
{
   Item* i_path = ctx->param(0);
   if( i_path == 0 || ! i_path->isString() )
   {
      throw paramError();
   }
   
   PathCarrier* pc = static_cast<PathCarrier*>(ctx->self().asInst());
   Item ret; 
   ret.setBoolean( pc->m_path.relativize( *i_path->asString() ) );
   ctx->returnFrame(ret);   
}


FALCON_DEFINE_METHOD_P1( ClassPath, canonicize )
{
   PathCarrier* pc = static_cast<PathCarrier*>(ctx->self().asInst());
   pc->m_path.canonicize();
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassPath, cwd )
{   
   String temp;
   Path::currentWorkDirectory( temp );
   ctx->returnFrame( temp );
}


}
}

/* end of path.cpp */
