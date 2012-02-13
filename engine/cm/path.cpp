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
#include <falcon/path.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/cm/uri.h>

namespace Falcon {
namespace Ext {


ClassPath::ClassPath():
   ClassUser("Path"),
   
   FALCON_INIT_PROPERTY( resource ),
   FALCON_INIT_PROPERTY( location ),
   FALCON_INIT_PROPERTY( fulloc ),
   FALCON_INIT_PROPERTY( file ),
   FALCON_INIT_PROPERTY( ext ),
   FALCON_INIT_PROPERTY( filext ),
   FALCON_INIT_PROPERTY( encoded ),
   
   FALCON_INIT_PROPERTY( wlocation ),
   FALCON_INIT_PROPERTY( wfulloc ),
   FALCON_INIT_PROPERTY( wencoded ),

   FALCON_INIT_METHOD( absolutize ),
   FALCON_INIT_METHOD( relativize ),
   FALCON_INIT_METHOD( canonicize ),
   FALCON_INIT_METHOD( cwd )
{
}

ClassPath::~ClassPath()
{}


void ClassPath::store( VMContext*, DataWriter* stream, void* instance ) const
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   stream->write(pc->m_path.encode());
}


void ClassPath::restore( VMContext*, DataReader* stream, void*& empty ) const
{
   String pathName;
   stream->read( pathName );
   PathCarrier* pc = new PathCarrier(carriedProps());
   pc->m_path.parse( pathName );
   empty = pc;
}


void* ClassPath::createInstance() const
{
   return new PathCarrier( carriedProps() );
}


bool ClassPath::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   PathCarrier* uc = static_cast<PathCarrier*>(instance);
   
   if ( pcount >= 1 )
   {
      Item& other = *ctx->opcodeParams(pcount);
      if( other.isString() )
      {        
         if( ! uc->m_path.parse( *other.asString() ) )
         {
            delete uc;
            throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC )
               .origin(ErrorParam::e_orig_mod) );
         }
      }
      else if( other.asClass() == this )
      {
         uc->m_path = static_cast<PathCarrier*>(other.asInst())->m_path;
      }
      else
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
               .extra( "S|Path" )
               .origin(ErrorParam::e_orig_mod) );
      }
      
   }
  
   return false;
}


void ClassPath::op_toString( VMContext* ctx, void* self ) const
{
   PathCarrier* uc = static_cast<PathCarrier*>(self);
   ctx->topData().setString( uc->m_path.encode() ); // garbages
}


//====================================================
// Properties.
//
   
FALCON_DEFINE_PROPERTY_SET_P( ClassPath, resource )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.resource(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, resource )
{
   return static_cast<PathCarrier*>(instance)->m_path.resource();
}


FALCON_DEFINE_PROPERTY_SET_P( ClassPath, location )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.location(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, location )
{
   return static_cast<PathCarrier*>(instance)->m_path.location();
}


FALCON_DEFINE_PROPERTY_SET_P( ClassPath, fulloc )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.fulloc(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, fulloc )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_path.getFullLocation(pc->m_fulloc);   
   return pc->m_fulloc;
}
   
  
FALCON_DEFINE_PROPERTY_SET_P( ClassPath, file )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.file(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, file )
{
   return static_cast<PathCarrier*>(instance)->m_path.file();
}
   

FALCON_DEFINE_PROPERTY_SET_P( ClassPath, ext )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.ext(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, ext )
{
   return static_cast<PathCarrier*>(instance)->m_path.ext();
}

FALCON_DEFINE_PROPERTY_SET_P( ClassPath, filext )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.fileext(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, filext )
{
   return static_cast<PathCarrier*>(instance)->m_path.fileext();
}


FALCON_DEFINE_PROPERTY_SET_P( ClassPath, encoded )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.parse(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, encoded )
{
   return static_cast<PathCarrier*>(instance)->m_path.encode();
}


FALCON_DEFINE_PROPERTY_SET_P( ClassPath, wlocation )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.location(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, wlocation )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_path.getWinLocation(pc->m_winLoc);   
   return pc->m_winLoc;
}


FALCON_DEFINE_PROPERTY_SET_P( ClassPath, wfulloc )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.fulloc(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, wfulloc )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_path.getFullWinLocation(pc->m_fullWinLoc);   
   return pc->m_fullWinLoc;
}


FALCON_DEFINE_PROPERTY_SET_P( ClassPath, wencoded )
{
   checkType( value.isString(), "S" );
   if( ! static_cast<PathCarrier*>(instance)->m_path.parse(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

FALCON_DEFINE_PROPERTY_GETS_P( ClassPath, wencoded )
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
