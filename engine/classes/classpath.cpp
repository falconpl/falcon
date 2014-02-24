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

/*#
 @class Path
 @brief Interface to abstract and concrete system path specifications
 @optparam p The path to be parsed
 @raise ParseError if the path is invalid

 The Path class handles abstract file names manipulation, as well as
 system-specific transformation of local relative path into absolute
 paths and the other way around.

 A Falcon path is represented according to RFC3986 URI specification:
 each element of the path is separated by the next by a forward
 slash character ('/'). In case an extra unit or phyisical local
 resource specificator is needed, it is the first element of the
 path, prefixed with a single slash '/' character and followed by a colon ':'.

 So, the file 'test.txt' under the location 'directory' in the location 'local',
 stored in the 'C' unit, will be fully represented as:

 @code
 /C:/local/directory/test.txt
 @endcode

 All the functions dealing with abstracts URI or local paths
 understand this format, which is widely compatible with POSIX specifications
 as well as with the RFC3986 standard specification.

 However, to simply generate path specifications that can be literally fed into
 MS-Windows programs (i.e. when generating batch files or invoking MS-Windows SDK
 functions directly), this class offers also utilities to represent the paths
 as required by MS-Windows systems.

 @note This class has type dignity and a numeric type id.

 @prop resource Get or set the local resource part (disk, device, etc.)
 @prop location Get or set the location part (prefix path to file)
 @prop fulloc Get or set the composition of the resource and location.
 @prop file Get or set the file component of the path
 @prop ext Get or set the extension component of the path (part past the last dot)
 @prop filext Get or set both the file and the extension component at once.
 @prop encoded Get or set the whole path, fully encoded.
 @prop wlocation Get the MS-Windows specific representation of a path.
 @prop wfulloc Get the MS-Windows specific representation of a path and its unit
 @prop wencoded Get a full MS-Windows specific representation of a path.
*/

/*#
@method absolutize Path
@brief Ensures that this path is absolute.
@optparam parent A parent directory to use for absolutization
@return This path object.
@raise IOError on system error reading the current working directory.

This method transform the current path in an absolute one, if it isn't
already absolute, by reading the current working directory from the
system and appending it to the local path. If a @b parent parameter is
provided, that is used instead.


*/

/*#
@method relativize Path
@brief Transforms an absolute path into a relative one.
@param parent The path to which this path is to be made relative to.
@return This path object.
@raise IOError on system error reading the current working directory.

This method transform the current path in a relative one, taking the
@b parent object and calculating a relative path that will lead from
that one to the absolute path stored in this object.

If necessary, relative parent entries ("../") will be added to
ascend into the relative parent paths.

*/

/*#
@method canonicize Path
@brief Removes redoundant and useless redirections in the path.
@return this same path object

This method removes all the entries pointing to the same location ("./"),
and the ones pointing to the parent location that are not strictly necessary.
for instance:
@code
   a/./b/../file => a/file
@endcode
*/


/*#
@method cwd Path
@brief (static) Returns the current work directory.
@return The current work directory for the host process.

The path object is not meant to be used to change the current work
directory of the host process; this task is performed by
separate modules (as the VFS or system-specific modules).
*/


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
   ctx->returnFrame( FALCON_GC_HANDLE( new String(temp)) );
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
            throw new ParseError( ErrorParam( e_malformed_uri, __LINE__, SRC )
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
