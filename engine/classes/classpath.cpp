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

#include <falcon/classes/classpath.h>
#include <falcon/itemid.h>

#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/path.h>
#include <falcon/stderrors.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/classes/classuri.h>

namespace Falcon {

   
FALCON_DECLARE_FUNCTION( absolutize, "parent:[S]" );
FALCON_DECLARE_FUNCTION( relativize, "parent:S" );
FALCON_DECLARE_FUNCTION( canonize, "" );
FALCON_DECLARE_FUNCTION( cwd, "" );
//==============================================================

class PathCarrier
{
public:
   Path m_path;
   uint32 m_mark;
   
   
   PathCarrier():
   m_mark(0)
   {}
   
   PathCarrier( const PathCarrier& other ):
      m_path( other.m_path ),
      m_mark(0)
   {}
   
   ~PathCarrier()
   {
   }

};

//====================================================
// Properties.
//
   
static void set_resource( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.resource(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_resource( const Class*, const String&, void* instance, Item& value )
{
   value = FALCON_GC_HANDLE( new String(static_cast<PathCarrier*>(instance)->m_path.resource()));
}


static void set_location( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.location(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_location( const Class*, const String&, void* instance, Item& value )
{
   value = FALCON_GC_HANDLE( new String(static_cast<PathCarrier*>(instance)->m_path.location()));
}


static void set_fulloc( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.fulloc(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}
  

static void get_fulloc( const Class*, const String&, void* instance, Item& value )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   String* loc = new String;
   pc->m_path.getFullLocation(*loc);   
   value = FALCON_GC_HANDLE(loc) ;
}

  
static void set_file( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.filename(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

   
static void get_file( const Class*, const String&, void* instance, Item& value )
{
   value = FALCON_GC_HANDLE( new String(static_cast<PathCarrier*>(instance)->m_path.filename()));
}


static void set_ext( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.ext(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

static void get_ext( const Class*, const String&, void* instance, Item& value )
{
   value = FALCON_GC_HANDLE( new String(static_cast<PathCarrier*>(instance)->m_path.ext()));
}


static void set_filext( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.file(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_filext( const Class*, const String&, void* instance, Item& value )
{
   value = FALCON_GC_HANDLE( new String(static_cast<PathCarrier*>(instance)->m_path.file()));
}


static void set_encoded( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.parse(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_encoded( const Class*, const String&, void* instance, Item& value )
{
   value = FALCON_GC_HANDLE( new String(static_cast<PathCarrier*>(instance)->m_path.encode()));
}


static void set_wlocation( const Class*, const String&, void* instance, const Item& value )
{   
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.location(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}

static void get_wlocation( const Class*, const String&, void* instance, Item& value )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   String* temp = new String;
   pc->m_path.getWinLocation(*temp);   
   value = FALCON_GC_HANDLE(temp);
}


static void set_wfulloc( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.fulloc(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_wfulloc( const Class*, const String&, void* instance, Item& value )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   String* temp = new String;
   pc->m_path.getFullWinLocation(*temp);   
   value = FALCON_GC_HANDLE( temp );
}


static void set_wencoded( const Class*, const String&, void* instance, const Item& value )
{
   if( ! value.isString() )
   {
      throw new AccessError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
         .extra("S") );
   }

   if( ! static_cast<PathCarrier*>(instance)->m_path.parse(*value.asString()) )
   {
      throw new ParamError( ErrorParam( e_malformed_uri, __LINE__, SRC ) );
   }
}


static void get_wencoded( const Class*, const String&, void* instance, Item& value )
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   String* temp = new String;
   pc->m_path.getWinFormat(*temp);   
   value = FALCON_GC_HANDLE( temp );
}


void Function_absolutize::invoke( VMContext* ctx, int32 )
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


void Function_relativize::invoke( VMContext* ctx, int32 )
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


void Function_canonize::invoke( VMContext* ctx, int32 )
{
   PathCarrier* pc = static_cast<PathCarrier*>(ctx->self().asInst());
   pc->m_path.canonicize();
   ctx->returnFrame();
}



void Function_cwd::invoke( VMContext* ctx, int32 )
{   
   String temp;
   Path::currentWorkDirectory( temp );
   ctx->returnFrame( temp );
}


//==============================================================


ClassPath::ClassPath():
   Class("Path", FLC_CLASS_ID_PATH)
{
   addProperty( "resource", &get_resource, &set_resource );
   addProperty( "location", &get_location, &set_location );
   addProperty( "fulloc", &get_fulloc, &set_fulloc );
   addProperty( "file", &get_file, &set_file );
   addProperty( "ext", &get_ext, &set_ext );
   addProperty( "filext", &get_filext, &set_filext );
   addProperty( "encoded", &get_encoded, &set_encoded );
   addProperty( "wlocation", &get_wlocation, &set_wlocation );
   addProperty( "wfulloc", &get_wfulloc, &set_wfulloc );
   addProperty( "wencoded", &get_wencoded, &set_wencoded );

   addMethod( new Function_absolutize );
   addMethod( new Function_relativize );
   addMethod( new Function_canonize );
   addMethod( new Function_cwd, true );
}

ClassPath::~ClassPath()
{}


void ClassPath::dispose( void* instance ) const
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   delete pc;
}

void* ClassPath::clone( void* instance ) const
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   return new PathCarrier( *pc );
}

void ClassPath::gcMarkInstance( void* instance, uint32 mark ) const
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   pc->m_mark = mark;
}

bool ClassPath::gcCheckInstance( void* instance, uint32 mark ) const
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   return pc->m_mark >= mark;
}


void ClassPath::store( VMContext*, DataWriter* stream, void* instance ) const
{
   PathCarrier* pc = static_cast<PathCarrier*>(instance);
   stream->write(pc->m_path.encode());
}


void ClassPath::restore( VMContext* ctx, DataReader* stream ) const
{
   String pathName;
   stream->read( pathName );
   PathCarrier* pc = new PathCarrier();
   try {
      pc->m_path.parse( pathName );
      ctx->pushData( Item( this, pc ) );
   }
   catch( ... ) {
      delete pc;
      throw;
   }
}


void* ClassPath::createInstance() const
{
   return new PathCarrier( );
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


}

/* end of path.cpp */
