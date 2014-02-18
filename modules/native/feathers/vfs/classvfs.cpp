/*
   FALCON - The Falcon Programming Language.
   FILE: classvfs.cpp

   Interface for script to Shared variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Mar 2013 00:25:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "modules/native/feathers/vfs/classvfs.cpp"

#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/stderrors.h>
#include <falcon/stdhandlers.h>
#include <falcon/uri.h>
#include <falcon/classes/classuri.h>
#include <falcon/vfsprovider.h>
#include <falcon/stream.h>

#include "vfs.h"
#include "classvfs.h"

namespace Falcon {
namespace Ext {

URI* ClassVFS::internal_get_uri( Item* i_uri, URI& tempURI, Module* )
{
   static Class* uriClass = Engine::instance()->stdHandlers()->uriClass();

   if( i_uri == 0 )
   {
      return 0;
   }

   if( i_uri->isString() )
   {
      if( ! tempURI.parse(*i_uri->asString()) )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra( "Invalid URI" ));
      }

      return &tempURI;
   }

   Class* cls; void* data;
   if( ! i_uri->asClassInst( cls, data )
     || ! cls->isDerivedFrom(uriClass)
     )
   {
      return 0;
   }

   URI* uricar = static_cast<URI*>(cls->getParentData( uriClass, data ));
   return uricar;
}


static VFSProvider* internal_get_provider( const Item& self, const String& scheme )
{
   VFSProvider* prov;
   if( self.isNil())
   {
      prov = Engine::instance()->vfs().getVFS( scheme );
      if( prov == 0 )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra( String("Unknown URI scheme ") + scheme ));
      }
   }
   else {
      prov = static_cast<VFSProvider*>( self.asInst() );
   }

   return prov;
}


namespace _classVFS {


/*#
   @class VFS
   @brief Virtual file system interface
   @param scheme The Virtual File System scheme used by this VFS instance
   @raise ParamError if the given scheme is not registered in the engine.

   The methods in this class can be invoked statically (directly from the
   VFS class) or from an instance.

   If they're invoked statically, their scheme is searched among the engine
   VFS provider each time. To operate on a virtual file system without specifying
   the VFS scheme each time, or just to check if the VFS is available, it is
   possible to create an instance of this class and then operate on that.

   @code
   // you can do...
   stream = VFS.open("myscheme://dirname/filename")

   // or...
   fs = VFS("myscheme")
   // ...
   stream = fs.open("/dirname/filename")
   @endcode
 */

FALCON_DECLARE_FUNCTION( construct ,"scheme:S" )
void Function_construct::invoke( Falcon::VMContext* ctx, int )
{
   Item* i_scheme = ctx->param(0);
   if( i_scheme == 0 || ! i_scheme->isString() )
   {
      throw paramError( __LINE__, SRC );
   }

   const String& scheme = *i_scheme->asString();
   VFSProvider* prov = Engine::instance()->vfs().getVFS( scheme );
   if( prov == 0 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra( String("Unknown URI scheme ") + scheme ));
   }

   // VFS are never decreffed. -- at least for now.
   ctx->self().setUser(methodOf(), prov);
   ctx->returnFrame(ctx->self());
}


/*#
 @method open VFS
 @brief Opens a Virtual File System entity.
 @param uri The VFS uri (string or URI entity) to be opened.
 @optparam mode Open mode.
 @optparam shmode Share mode.
 @return On success a new stream.
 @raise IoError on error.

 The open mode could be an (bitwise-or) combination of the following:
 - @b O_RD: Read only
 - @b O_WR: Write only
 - @b O_APPEND: Set the file pointer at end
 - @b O_TRUNC:  Truncate

Share modes are a combination of the following.
 - @b SH_NR: Shared read
 - @b SH_NW: Shared write

  The real meaning of the settings depends on the final
  virtual file system driver.

  By default the stream is opened as O_RD.
 */
FALCON_DECLARE_FUNCTION( open ,"uri:S|URI,mode:[N],shmode:[N]" )
void Function_open::invoke( Falcon::VMContext* ctx, int )
{
   Item* i_uri = ctx->param(0);
   Item* i_mode = ctx->param(1);
   Item* i_shmode = ctx->param(2);

   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0
        || (i_mode != 0 && (i_mode->isInteger()||i_mode->isNil()))
        || (i_shmode != 0 && i_shmode->isInteger())
      )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   VFSProvider::OParams op;

   if( i_mode != 0 && ! i_mode->isNil() )
   {
      op = VFSProvider::OParams((int32)i_mode->forceInteger());
   }
   else
   {
      op.rdOnly();
   }

   if( i_shmode != 0 )
   {
      int64 shMode = i_shmode->asInteger();
      if( shMode & VFSProvider::OParams::e_sflag_nr )
      {
         op.shNoRead();
      }
      if( shMode & VFSProvider::OParams::e_sflag_nw )
      {
         op.shNoWrite();
      }
   }

   VFSProvider* iface = internal_get_provider( ctx->self(), uri->scheme() );
   Stream* stream = iface->open( *uri, op );
   stream->shouldThrow(true);
   ctx->returnFrame( FALCON_GC_HANDLE( stream ) );
}


/*#
 @method create VFS
 @brief Creates a new virtual file system entity.
 @param uri The VFS uri (string or URI entity) to be opened.
 @optparam mode Create mode.
 @return On success a new stream.
 @raise IoError on error.

 - @b O_RD: Read only
 - @b O_WR: Write only
 - @b O_APPEND: Set the file pointer at end
 - @b O_TRUNC:  Truncate

 - @b SH_NR: Shared read
 - @b SH_NW: Shared write

  - C_NOOVR: Do not overwrite the file if it already exists.
  - C_NOSTREAM: Do not create a steram on error (implied when throwing an I/O error).

  The real meaning of the settings depends on the final
  virtual file system driver.

  By default the stream is opened as O_RD.

 */

FALCON_DECLARE_FUNCTION( create ,"uri:S|URI,mode:[N]" )
void Function_create::invoke( Falcon::VMContext* ctx, int )
{
   Item* i_uri = ctx->param(0);
   Item* i_mode = ctx->param(1);

   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0
        || (i_mode != 0 && i_mode->isInteger())
      )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   VFSProvider::CParams op;
   if( i_mode != 0 )
   {
      op = VFSProvider::CParams(i_mode->forceInteger());
   }
   else
   {
      op.wrOnly();
   }


   VFSProvider* iface = internal_get_provider( ctx->self(), uri->scheme() );
   Stream* stream = iface->create( *uri, op );
   stream->shouldThrow(true);
   ctx->returnFrame( FALCON_GC_HANDLE( stream ) );
}


/*#
   @method mkdir VFS
   @brief Creates a directory on a virtual file system.
   @param dirname The name of the directory to be created.
   @optparam withParents Create also the full pat to the given directory.
   @raise IoError on system error.

   On success, this function creates the given directory with normal
   attributes.

   It is possible to specify both a relative or absolute path; both
   the relative and absolute path can contain a subtree specification,
   that is, a set of directories separated by forward slashes, which
   lead to the directory that should be created. For example:

   @code
      VFS.mkdir( "top/middle/bottom" )
   @endcode

   instructs @b mkdir to create the directory bottom in a subdirectory
   "middle", which should already exist. Passing @b true as second parameter,
   mkdir will also try to create directories leading to the final destination,
   if they are missing.
*/
FALCON_DECLARE_FUNCTION( mkdir ,"uri:S|URI,withParents:[B]" )
void Function_mkdir::invoke( Falcon::VMContext* ctx, int )
{
   Item* i_uri = ctx->param(0);

   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0 )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   Item* i_parent = ctx->param(1);
   bool bParent = i_parent != 0 && i_parent->isTrue();

   VFSProvider* iface = internal_get_provider( ctx->self(), uri->scheme() );
   iface->mkdir(*uri, bParent);
   ctx->returnFrame();
}

/*#
   @method erase VFS
   @brief Removes a file from the target virtual file system.
   @param uri The URI of the file to be removed (string or URI instance).
   @raise IoError on system error.

   On failure, an IoError is raised.
*/

FALCON_DECLARE_FUNCTION( erase ,"uri:S|URI" )
void Function_erase::invoke( Falcon::VMContext* ctx, int )
{
   Item* i_uri = ctx->param(0);

   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0 )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   VFSProvider* iface = internal_get_provider( ctx->self(), uri->scheme() );
   iface->erase(*uri);
   ctx->returnFrame();
}


/*#
   @method move VFS
   @brief Moves a file in the same virtual file system.
   @param source The URI of the file to be moved (string or URI instance).
   @param dest The URI of the destination (string or URI instance).
   @raise IoError on system error.

   This function actually renames a file. It will typically fail if
   the source and the destination are on two different VFS.

   On failure, an IoError is raised.
*/

FALCON_DECLARE_FUNCTION( move ,"soruce:S|URI,dest:S|URI" )
void Function_move::invoke( Falcon::VMContext* ctx, int )
{
   Item* i_source = ctx->param(0);
   Item* i_dest   = ctx->param(1);

   URI tmpUri1, tmpUri2;
   URI* uri1, *uri2;
   if (
         (uri1 = ClassVFS::internal_get_uri( i_source, tmpUri1, m_module )) == 0
         || (uri2 = ClassVFS::internal_get_uri( i_dest, tmpUri2, m_module )) == 0
      )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   VFSProvider* iface = internal_get_provider( ctx->self(), uri1->scheme() );
   iface->move(*uri1, *uri2);
   ctx->returnFrame();
}


/*#
 @method fileType VFS
 @brief Determines the logical type of a file on a Virtual File System.
 @param uri The target file.
 @optparam deref If true, dereference logical links automatically as the target file type.
 @return One of the possible logical type values

 The return values will be one of the following constants defined in the VFS class:

 - NOT_FOUND
 - UNKNOWN
 - NORMAL
 - DIR
 - LINK
 - DEVICE
 - SOCKET

 */

FALCON_DECLARE_FUNCTION( fileType ,"uri:S|URI,deref:[B]" )
void Function_fileType::invoke( Falcon::VMContext* ctx, int )
{
   Item* i_uri   = ctx->param(0);

   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0 )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   Item* i_deref = ctx->param(1);
   bool bDeref = i_deref != 0 && i_deref->isTrue();

   VFSProvider* iface = internal_get_provider( ctx->self(), uri->scheme() );
   FileStat::t_fileType ft = iface->fileType(*uri, bDeref);
   ctx->returnFrame( (int64) ft );
}


/*#
 @method readStats VFS
 @brief Reads the generic file stats associated with a given URI.
 @param uri The target file.
 @optparam deref If true, dereference logical links automatically as the target file type.
 @return A @a FileStat instance on success, nil if the given file doesn't exist.

 This method creates a new FileStat instance each time it is invoked. When repeatedly
 asking the stats of multiple files, it's preferable to use the @a FileStat.read to
 use the same FileStat instance multiple times.

 */

FALCON_DECLARE_FUNCTION( readStats ,"uri:S|URI,deref:[B]" )
void Function_readStats::invoke( Falcon::VMContext* ctx, int )
{
   static Class* statCls = methodOf()->module()->getClass("FileStat");
   fassert( statCls != 0 );

   Item* i_uri   = ctx->param(0);
   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0 )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   Item* i_deref = ctx->param(1);
   bool bDeref = i_deref != 0 && i_deref->isTrue();

   VFSProvider* iface = internal_get_provider( ctx->self(), uri->scheme() );
   FileStat fs;
   if( iface->readStats( *uri, fs, bDeref) )
   {
      ctx->returnFrame( FALCON_GC_STORE(statCls, new FileStat(fs) ) );
   }
   else {
      ctx->returnFrame();
   }
}

/*#
 @method openDir VFS
 @brief Open a virtual directory handle to read the contents of a directory.
 @param uri The directory to be opened.
 @return A @a Directory instance.
 @raise IoError if the directory cannot be accessed (or is not a directory).
*/

FALCON_DECLARE_FUNCTION( openDir ,"uri:S|URI" )
void Function_openDir::invoke( Falcon::VMContext* ctx, int )
{
   static Class* dirCls = methodOf()->module()->getClass("Directory");
   fassert( dirCls != 0 );

   Item* i_uri = ctx->param(0);
   URI tmpUri;
   URI* uri;
   if ( (uri = ClassVFS::internal_get_uri( i_uri, tmpUri, m_module )) == 0 )
   {
      ctx->raiseError(paramError( __LINE__, SRC ) );
      return;
   }

   VFSProvider* iface = internal_get_provider( ctx->self(), uri->scheme() );
   Directory* dir = iface->openDir( *uri );
   fassert( dir != 0 ); // or we should have raised.
   if( dir != 0 )
   {
      // read the first entry now, dir->read() expects it
      dir->read(dir->next());
      ctx->returnFrame( FALCON_GC_STORE(dirCls, dir) );
      return;
   }

   // just in case.
   ctx->returnFrame();
}

}

/*# @property protocol VFS
 @brief Gets the protocol name associated with this virtual file system.
 */
static void get_protocol( const Class*, const String&, void *instance, Item& value )
{
   VFSProvider* prov = static_cast<VFSProvider*>(instance);
   value = FALCON_GC_HANDLE(new String(prov->protocol()));
}

//=========================================================
//
//=========================================================


ClassVFS::ClassVFS():
         Class("VFS")
{
   setConstuctor( new _classVFS::Function_construct );

   addMethod( new _classVFS::Function_open, true);
   addMethod( new _classVFS::Function_create, true);
   addMethod( new _classVFS::Function_mkdir, true);
   addMethod( new _classVFS::Function_erase, true);
   addMethod( new _classVFS::Function_move, true);
   addMethod( new _classVFS::Function_readStats, true);
   addMethod( new _classVFS::Function_fileType, true);
   addMethod( new _classVFS::Function_openDir, true);

   addProperty( "protocol", &get_protocol );

   addConstant( "NOT_FOUND", FileStat::_notFound );
   addConstant( "UNKNOWN", FileStat::_unknown );
   addConstant( "NORMAL", FileStat::_normal );
   addConstant( "DIR", FileStat::_dir );
   addConstant( "LINK", FileStat::_link );
   addConstant( "DEVICE", FileStat::_device );
   addConstant( "SOCKET", FileStat::_socket );
}


ClassVFS::~ClassVFS()
{
}

void ClassVFS::dispose( void* ) const
{
   // FOR NOW, do nothing
}

void* ClassVFS::clone( void* instance ) const
{
   // FOR NOW, just return a copy of the item
   return instance;
}

void* ClassVFS::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

}
}

/* end of classvfs.cpp */
