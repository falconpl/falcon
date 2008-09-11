/*
   FALCON - The Falcon Programming Language.
   FILE: vfsprovider.h

   Support for directory oriented operations in unix systems
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Sep 2008 08:58:33 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Support for directory oriented operations in unix systems.
*/

#ifndef flc_vsf_provider_H
#define flc_vfs_provider_H

#include <falcon/filestat.h>
#include <falcon/dir_sys.h>
#include <falcon/string.h>

namespace Falcon {
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
FLC_DYN_CLASS class VSFProvider: public BaseAlloc
{
   String m_servedProto;

protected:
   VFSProvider( const String &name ):
      m_name( name )
   {}

public:

   /** Open Parameters.
      Parameters for opening a stream on the provider.
      Subclasses may overload this class to provide VFS
      specific open-flags.
   */
   class OParams {
      uint32 m_flags;
      friend class VSFProvider;

   public:
      OParams():
         m_flags(0)
      {}
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
   */

   class CParams {
      uint32 m_flags;
      friend class VFSProvider;
   public:
      CParams():
         m_flags(0)
      {}

      /** Indicates that the created file will be a directory. */
      CParams &setDir() { m_flags |= 1; return *this; }
      bool isDir() const { return (m_flags & 1) == 1; }
   };

   /** Just an inline for opening file with default parameters.
      Default parameters are "read only, full sharing".
   */
   inline Stream *open( const String &path ) {
      return open( path, OParams() );
   }

   /** Open a file. */
   virtual Stream *open( const String &path, const OParams &p )=0;

   inline bool create( const String &path ) {
      return create( path, CParams() );
   }


   virtual bool create( const String &path, const CParams &p )=0;

   virtual DirEntry* openDir( const String &path )=0;
   virtual bool readStats( FileStat &s )=0;
   virtual bool writeStats( const FileStats &s )=0;

   virtual int64 getLastFsError()=0;
   virtual IOError *getLastError()=0;
};

}

/* end of vsfprovider.h */
