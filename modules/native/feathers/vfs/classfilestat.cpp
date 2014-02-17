/*
   FALCON - The Falcon Programming Language.
   FILE: classfilestat.cpp

   Falcon core module -- Structure holding information on files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Mar 2013 00:25:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "modules/native/feathers/vfs/classfilestat.cpp"

#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/stderrors.h>
#include <falcon/uri.h>
#include <falcon/classes/classuri.h>
#include <falcon/filestat.h>
#include <falcon/stream.h>

#include "vfs.h"
#include "classfilestat.h"
#include "classvfs.h"

namespace Falcon {
namespace Ext {


/*#
   @class FileStat
   @optparam path If given, the filestats will be initialized with stats of the given file.
   @raise IoError if @b path is given but not found.
   @brief Class holding informations on system files.

   The FileStat class holds informations on a single directory entry.

   It is possible to pass a @b path parameter, in which case, if the given file is found,
   the contents of this class is filled with the stat data from the required file, otherwise
   an IoError is raised. The @a FileStat.read method would search for the required file
   without raising in case it is not found, so if it preferable not to raise on failure
   (i.e. because searching the most fitting of a list of possibly existing files), it is
   possible to create the FileStat object without parameters and the use the @b read method
   iteratively.

   @prop atime Last access time, expressed in milliseconds from epoch
   @prop ctime Creation time or last attribute change time, in milliseconds from epoch
   @prop mtime Last modify time, expressed in milliseconds from epoch
   @prop size File size.

   @note Times can be easily transformed in a @a TimeStamp class instance by setting the
   @a TimeStamp.msSinceEpoch member.
*/

/*# @property ftype FileStat
   @brief Type of the file.

   Can be one of the following constants (declared in this class):

   - NORMAL
   - DIR
   - PIPE
   - LINK
   - DEVICE
   - SOCKET
   - UNKNOWN
*/

namespace _classFileStat {

/*#
 @method read FileStat
 @brief Reads the generic file stats associated with a given URI.
 @param uri The target file.
 @optparam deref If true, dereference logical links automatically as the target file type.
 @return True if the requested file was found, false otherwise.
 @raise IoError on error while accessing the resource.

 This method overwrites this FileStat instance with the new stats
 for the given file, if it's found.
*/

FALCON_DECLARE_FUNCTION( read ,"uri:S|URI,deref:[B]" )
void Function_read::invoke( Falcon::VMContext* ctx, int )
{

   Item* i_uri   = ctx->param(0);
   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0)
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   Item* i_deref = ctx->param(1);
   bool bDeref = i_deref != 0 && i_deref->isTrue();

   FileStat* fs = static_cast<FileStat*>(ctx->self().asInst());
   bool bResult = Engine::instance()->vfs().readStats( *uri, *fs, bDeref);
   ctx->returnFrame( Item().setBoolean(bResult) );
}
}

static void set_atime( const Class*, const String&, void *instance, const Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_prop_value, .extra("N"));
   }

   fs->atime(value.forceInteger());
}

static void get_atime( const Class*, const String&, void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value = fs->atime();
}


static void set_mtime( const Class*, const String&, void *instance, const Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_prop_value, .extra("N"));
   }

   fs->mtime(value.forceInteger());
}

static void get_mtime( const Class*, const String& , void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value = fs->mtime();
}

static void set_ctime( const Class*, const String&, void *instance, const Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_prop_value, .extra("N"));
   }

   fs->ctime(value.forceInteger());
}

static void get_ctime( const Class*, const String&, void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value = fs->ctime();
}

static void set_type( const Class*, const String&, void *instance, const Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_prop_value, .extra("N"));
   }

   fs->type(static_cast<FileStat::t_fileType>(value.forceInteger()));
}

static void get_type( const Class*, const String&, void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value = (int64) fs->type();
}

static void set_size( const Class*, const String&, void *instance, const Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_prop_value, .extra("N"));
   }

   fs->size(value.forceInteger());
}

static void get_size( const Class*, const String&, void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value = (int64) fs->size();
}

static void get_exists( const Class*, const String&, void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value.setBoolean(fs->exists());
}

static void get_isDirectory( const Class*, const String&, void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value.setBoolean(fs->isDirectory());
}

static void get_isFile( const Class*, const String&, void *instance, Item& value )
{
   FileStat* fs = static_cast<FileStat*>(instance);
   value.setBoolean(fs->isFile());
}

//=========================================================
//
//=========================================================


ClassFileStat::ClassFileStat():
         Class("FileStat")
{
   addConstant( "NOT_FOUND", FileStat::_notFound );
   addConstant( "UNKNOWN", FileStat::_unknown );
   addConstant( "NORMAL", FileStat::_normal );
   addConstant( "DIR", FileStat::_dir );
   addConstant( "LINK", FileStat::_link );
   addConstant( "DEVICE", FileStat::_device );
   addConstant( "SOCKET", FileStat::_socket );

   setConstuctor( new _classFileStat::Function_read );
   addMethod( new _classFileStat::Function_read );

   addProperty("atime", &get_atime, &set_atime );
   addProperty("mtime", &get_mtime, &set_mtime );
   addProperty("ctime", &get_ctime, &set_ctime );
   addProperty("size", &get_size, &set_size );
   addProperty("type", &get_type, &set_type );
   addProperty("exists", &get_exists );
   addProperty("isDirectory", &get_isDirectory );
   addProperty("isFile", &get_isFile );
}


ClassFileStat::~ClassFileStat()
{
}

void ClassFileStat::dispose( void* instance ) const
{
   FileStat* fs = static_cast<FileStat*>(instance);
   delete fs;
}

void* ClassFileStat::clone( void* instance ) const
{
   FileStat* fs = static_cast<FileStat*>(instance);
   return new FileStat(*fs);
}

void* ClassFileStat::createInstance() const
{
   return new Falcon::FileStat;
}

void ClassFileStat::gcMarkInstance( void* instance, uint32 mark ) const
{
   FileStat* fs = static_cast<FileStat*>(instance);
   fs->gcMark( mark );
}

bool ClassFileStat::gcCheckInstance( void* instance, uint32 mark ) const
{
   FileStat* fs = static_cast<FileStat*>(instance);
   return fs->currentMark() >= mark;
}


}
}

/* end of classfilestat.cpp */
