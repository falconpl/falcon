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

#ifndef FALCON_VFSPROVIDER_H
#define FALCON_VFSPROVIDER_H

#include <falcon/setup.h>
#include <falcon/filestat.h>
#include <falcon/directory.h>
#include <falcon/string.h>
#include <falcon/uri.h>

namespace Falcon {

class Error;
class Stream;

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
class FALCON_DYN_CLASS VFSProvider
{
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
      OParams( uint32 flags = 0 ):
         m_oflags( flags ),
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
      friend class VFSProvider;

   public:
      CParams():
         m_cflags(0)
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
   };

   inline const String& protocol() const { return m_servedProto; }

   /** Just an inline for opening file with default parameters.
      Default parameters are "read only, full sharing".
   */
   virtual inline Stream *openRO( const URI &uri ) {
      return open( uri, OParams() );
   }

   /** Open a file. */
   virtual Stream* open( const URI &uri, const OParams &p )=0;

   inline Stream* create( const URI &uri ) {
      return create( uri, CParams() );
   }
   
   virtual Stream *create( const URI &uri, const CParams &p )=0;
   virtual Directory* openDir( const URI &uri )=0;

   virtual void mkdir( const URI &uri, bool bWithParents = true )=0;
   virtual void erase( const URI &uri )=0;
   /** Gets the stats of a given file.
      \param uri the file of which to get the stats.
      \param s The stats where to store the stats.
      \param delink if true, resolve symbolic links before returning the file stats.
      \return true if the file is found, false if it doesn't exists.
      \throw IOError if the the stats of an existing file cannot be read.
    */
   virtual bool readStats( const URI &uri, FileStat &s, bool delink = true )=0;

   /** Checks if a file exists, and in that case, returns the type of the file. 
      \param uri the file of which to get thes stats.
      \param delink if true, resolve symbolic links before returning the file stats.
      \return The file type as it would be returned in the file stats.
    */
   virtual FileStat::t_fileType fileType( const URI& uri, bool delink = true )=0;

   virtual void move( const URI &suri, const URI &duri ) = 0;
protected:
   VFSProvider( const String &name ):
      m_servedProto( name )
   {}

private:
   String m_servedProto;

};
}

#endif

/* end of vsfprovider.h */
