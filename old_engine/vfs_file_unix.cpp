/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_file.cpp

   VSF provider for physical file system on the host system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 12 Sep 2008 21:47:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vfs_file.h>
#include <falcon/error.h>
#include <falcon/fstream_sys_unix.h>
#include <falcon/dir_sys_unix.h>
#include <falcon/sys.h>
#include <falcon/autocstring.h>
#include <falcon/streambuffer.h>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <stdio.h>
#include <fcntl.h>

namespace Falcon {

// we don't use filesystem data.
VFSFile::VFSFile():
   VFSProvider( "file" ),
   m_fsdata(0)
{}


VFSFile::~VFSFile()
{
}


Stream *VFSFile::open( const URI& uri, const OParams &p )
{
   int omode = paramsToMode( p );

   // todo: do something about share mode
   AutoCString cfilename( uri.path() );

   errno = 0;
   int handle = ::open( cfilename.c_str(), omode );
   if ( handle >= 0 )
   {
      UnixFileSysData *ufd = new UnixFileSysData( handle, 0 );
      return new StreamBuffer( new FileStream( ufd ) );
   }

   return 0;
}


Stream *VFSFile::create( const URI& uri, const CParams &p, bool &bSuccess )
{
   int omode = paramsToMode( p );

   if ( p.isNoOvr() )
      omode |= O_EXCL;

   //TODO: something about sharing
   AutoCString cfilename( uri.path() );
   errno=0;

   umask( 0000 );
   int handle = ::open( cfilename.c_str(), O_CREAT | omode, p.createMode() );

   if ( handle >= 0 ) {
      bSuccess = true;
      if ( ! p.isNoStream() )
      {
         UnixFileSysData *ufd = new UnixFileSysData( handle, 0 );
         return new StreamBuffer( new FileStream( ufd ) );
      }
      else
         return 0;
   }

   bSuccess = false;
   return 0;
}


DirEntry* VFSFile::openDir( const URI& uri )
{
   AutoCString filename( uri.path() );

   DIR *dir = ::opendir( filename.c_str() );
   if ( dir == 0 ) {
      return 0;
   }

   return new DirEntry_unix( uri.path(), dir );
}


bool VFSFile::readStats( const URI& uri, FileStat &s )
{
   return Sys::fal_stats( uri.path(), s );
}


bool VFSFile::writeStats( const URI& uri, const FileStat &s )
{
   // TODO: write contents
   return false;
}

bool VFSFile::chown( const URI &uri, int uid, int gid )
{
   AutoCString filename( uri.path() );
   return ::chown( filename.c_str(), uid, gid ) == 0;
}


bool VFSFile::chmod( const URI &uri, int mode )
{
   AutoCString filename( uri.path() );
   return ::chmod( filename.c_str(), mode ) == 0;
}

bool VFSFile::link( const URI &uri1, const URI &uri2, bool bSymbolic )
{
   // TODO
   return false;
}


bool VFSFile::unlink( const URI &uri )
{
   AutoCString filename( uri.path() );
   return ::unlink( filename.c_str() ) == 0;
}

bool VFSFile::move( const URI &suri, const URI &duri )
{
   AutoCString filename( suri.path() );
   AutoCString dest( duri.path() );
   return ::rename( filename.c_str(), dest.c_str() ) == 0;
}


bool VFSFile::mkdir( const URI &uri, uint32 mode )
{
   AutoCString filename( uri.path() );
   return ::mkdir( filename.c_str(), mode ) == 0;
}


bool VFSFile::rmdir( const URI &uri )
{
   AutoCString filename( uri.path() );
   return ::rmdir( filename.c_str() ) == 0;
}


int64 VFSFile::getLastFsError()
{
   return (int64) errno;
}


Error *VFSFile::getLastError()
{
   if( errno != 0 )
   {
      IoError *e = new IoError( e_io_error );
      e->systemError( errno );
      return e;
   }

   return 0;
}


}

/* end of vsf_file.cpp */
