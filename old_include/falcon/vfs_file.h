/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_file.h

   VSF provider for physical file system on the host system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 12 Sep 2008 21:47:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   VSF provider for physical file system on the host system.
*/

#ifndef flc_vfs_file_H
#define flc_vfs_file_H

#include <falcon/vfsprovider.h>
#include <fcntl.h>

namespace Falcon {

/** VSF provider for physical file system on the host system.
   This class wraps the "file" URI provider and is implemented
   with different cpp files on different host systems.
*/
class FALCON_DYN_CLASS VFSFile: public VFSProvider
{
protected:
   void *m_fsdata;


   inline int paramsToMode( const OParams &p )
   {
      int omode = 0;

      if ( p.isRdwr() )
         omode = O_RDWR;
      else if ( p.isRdOnly() )
         omode = O_RDONLY;
      else
         omode = O_WRONLY;

      if( p.isTruncate() )
         omode |= O_TRUNC;

      if( p.isAppend() )
         omode |= O_APPEND;

      return omode;
   }

public:
   VFSFile();
   virtual ~VFSFile();

   virtual Stream* open( const URI &uri, const OParams &p );
   virtual Stream* create( const URI &uri, const CParams &p, bool &bSuccess );
   virtual DirEntry* openDir( const URI &uri );
   virtual bool readStats( const URI &uri, FileStat &s );
   virtual bool writeStats( const URI &uri, const FileStat &s );

   virtual bool chown( const URI &uri, int uid, int gid );
   virtual bool chmod( const URI &uri, int mode );

   virtual bool link( const URI &uri1, const URI &uri2, bool bSymbolic );
   virtual bool unlink( const URI &uri );

   virtual bool mkdir( const URI &uri, uint32 mode );
   virtual bool rmdir( const URI &uri );
   virtual bool move( const URI &suri, const URI &duri );

   virtual int64 getLastFsError();
   virtual Error *getLastError();
};

}

#endif

/* end of vsf_file.h */
