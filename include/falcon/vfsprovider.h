/*
   FALCON - The Falcon Programming Language.
   FILE: vfsprovider.h

   Generic provider of file system abstraction.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Sep 2008 08:58:33 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Generic provider of file system abstraction.
*/

#ifndef flc_vfs_provider_H
#define flc_vfs_provider_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/filestat.h>
#include <falcon/dir_sys.h>
#include <falcon/string.h>
#include <falcon/uri.h>

namespace Falcon {

class Error;

/** Base class for Falcon Virtual File System Providers.
   VFS providers are singletons containing virtual
   pure functions (function vectors) meant to give
   information about a particular filesystem, or
   factory functions generating VFS related objects
   as streams and directory handlers.

   VFS handler respond to a single protocol specification.
   When the VM receives a request to open an URI (be it a
   file or a specific directory) it parses the given
   uri and finds an appropriate VFS provider for that
   kind of resource.
*/
class FALCON_DYN_CLASS VFSProvider: public BaseAlloc
{
   String m_servedProto;

protected:
   VFSProvider( const String &name ):
      m_servedProto( name )
   {}

public:
   virtual ~VFSProvider();

   /** Open Parameters.
      Parameters for opening a stream on the provider.
      Subclasses may overload this class to provide VFS
      specific open-flags.
   */
   class OParams {
      uint32 m_oflags;
      uint32 m_shflags;
      friend class VSFProvider;

   public:
      OParams():
         m_oflags(0),
         m_shflags(0)
      {}

      OParams& rdOnly() { m_oflags |= 0x1; return *this; }
      bool isRdOnly() const { return (m_oflags & 0x1) == 0x1; }

      OParams& wrOnly() { m_oflags |= 0x2; return *this; }
      bool isWrOnly() const { return (m_oflags & 0x2) == 0x2; }

      OParams& rdwr() { m_oflags |= 0x3; return *this; }
      bool isRdwr() const { return (m_oflags & 0x3) == 0x3; }

      /** Open the file for append.
         File pointer is moved to the end of file at open.
         (Some FS guarantee also moving the file pointer at end of file
         after each write).
      */
      OParams& append() { m_oflags |= 0x4; return *this; }
      bool isAppend() const { return (m_oflags & 0x4) == 0x4; }

      /** If the file exists, it is truncated.

         Can be specified also when creating a file. In that case,
         truncating a file causes it's modification time to be changed,
         but all its other stats (as owner, security access, creation date, etc.)
         are left untouched.
      */
      OParams& truncate() { m_oflags |= 0x8; return *this; }
      bool isTruncate() const { return (m_oflags & 0x8) == 0x8; }

      OParams& shNoRead() { m_shflags |= 0x1; return *this; }
      bool isShNoRead() const { return (m_shflags & 0x1) == 0x1; }

      OParams& shNoWrite() { m_shflags |= 0x2; return *this; }
      bool isShNoWrite() const { return (m_shflags & 0x2) == 0x2; }

      OParams& shNone() { m_shflags |= 0x3; return *this; }
      bool isShNone() const { return (m_shflags & 0x3) == 0x3; }
   };

   /** Create Paramenter.

      Parameters for creating a file on the provider.
      Subclasses may overload this class to provide VFS
      specific create-flags.

      Note that the "create" operation is also entitled
      to create a directory on the VFS, if the appropriate
      flag is set.

      The class is used for variable parameters idiom
      in the VFSProvider::create() method.

      Read/write and share modes are inherited from
      open settings.
   */

   class CParams: public OParams
   {
      uint32 m_cflags;
      uint32 m_cmode;
      friend class VFSProvider;

   public:
      CParams():
         m_cflags(0),
         m_cmode( 0644 )
      {}

      /** Fail if the file exists.
         If the file exists and none of append() or truncate() options are specified,
         the operation fails.

         The subsystem is bound to return a nonzero value from getLastFsError() if
         returning faulty from a this operation.
      */
      CParams& noOvr() { m_cflags |= 0x1; return *this; }
      bool isNoOvr() const { return (m_cflags & 0x1) == 0x1; }

      /** Avoid returning an open stream to the caller.
         Usually, if create() is successful an open stream
         is returned. Conversely, if this flag is set, the create
         function will return 0, eventually closing immediately the
         handle to the file in those systems with "open creating" semantics.
      */
      CParams& noStream() { m_cflags |= 0x2; return *this; }
      bool isNoStream() const { return (m_cflags & 0x2) == 0x2; }

      CParams& createMode( uint32 cm ) { m_cmode = cm; return *this; }
      uint32 createMode() const { return m_cmode; }
   };

   inline const String& protocol() const { return m_servedProto; }

   /** Just an inline for opening file with default parameters.
      Default parameters are "read only, full sharing".
   */
   inline Stream *open( const URI &uri ) {
      return open( uri, OParams() );
   }

   /** Open a file. */
   virtual Stream* open( const URI &uri, const OParams &p )=0;

   inline Stream* create( const URI &uri ) {
      bool dummy;
      return create( uri, CParams(), dummy );
   }

   inline Stream* create( const URI& uri, bool &bSuccess ) {
      return create( uri, CParams(), bSuccess );
   }

   inline Stream* create( const URI& uri, const CParams &p ) {
      bool dummy;
      return create( uri, p, dummy );
   }

   virtual bool link( const URI &uri1, const URI &uri2, bool bSymbolic )=0;
   virtual bool unlink( const URI &uri )=0;

   virtual Stream *create( const URI &uri, const CParams &p, bool &bSuccess )=0;

   virtual DirEntry* openDir( const URI &uri )=0;

   virtual bool mkdir( const URI &uri, uint32 mode )=0;
   virtual bool rmdir( const URI &uri )=0;
   virtual bool move( const URI &suri, const URI &duri )=0;

   virtual bool readStats( const URI &uri, FileStat &s )=0;
   virtual bool writeStats( const URI &uri, const FileStat &s )=0;

   virtual bool chown( const URI &uri, int uid, int gid )=0;
   virtual bool chmod( const URI &uri, int mode )=0;

   /** Get an integer representing the last file system specific error.
      The semantic of this number may be different on different VFS,
      but in all the VFS a return value of 0 is granted to indicate that
      the last operation performed was succesful.

      Also, the returned error code must be made thread specific or otherwise
      reentrant/interlocked.
   */
   virtual int64 getLastFsError()=0;

   /** Wraps the last system error into a suitable Falcon Error.
      If getLastFsError() returns 0, then this method will return
      0 too.
   */
   virtual Error *getLastError()=0;
};
}

#endif

/* end of vsfprovider.h */
