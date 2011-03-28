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

#ifndef FALCON_VFS_FILE_H
#define FALCON_VFS_FILE_H

#include <falcon/vfsprovider.h>
#include <fcntl.h>

namespace Falcon {

/** VSF provider for physical file system on the host system.
   This class wraps the "file" URI provider and is implemented
   with different cpp files on different host systems.
*/
class FALCON_DYN_CLASS VFSFile: public VFSProvider
{
public:

   VFSFile();
   virtual ~VFSFile();

   virtual Stream* open( const URI &uri, const OParams &p );
   virtual Stream* create( const URI &uri, const CParams &p );
   virtual Directory* openDir( const URI &uri );
   virtual bool readStats( const URI &uri, FileStat &s );
   FileStat::t_fileType fileType( const URI& uri );

   virtual void mkdir( const URI &uri, bool bCreateParent=true );
   virtual void erase( const URI &uri );
   virtual void move( const URI &suri, const URI &duri );

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
};

}

#endif

/* end of vsf_file.h */
