/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_file_posix.cpp

   VSF provider for physical file system on the host system (POSIX).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 12 Sep 2008 21:47:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/vfs_file.h>
#include <falcon/error.h>
#include <falcon/sys.h>
#include <falcon/autocstring.h>
#include <falcon/streambuffer.h>
#include <falcon/fstream.h>
#include <falcon/ioerror.h>
#include <falcon/filestat.h>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <stdio.h>
#include <fcntl.h>


namespace Falcon {

class FALCON_DYN_CLASS Directory_file: public Directory
{
public:
   Directory_file( const URI* location ):
      Directory(location)
   {
         AutoCString loc(location.get());
         m_dir = ::opendir( loc.c_str() );
   }

   virtual ~Directory_file();
   virtual void close();
   virtual bool read( String& tgt );
   
private:
   DIR* m_dir;
   struct dirent m_de;
};

Directory_file::~Directory_file()
{}


void Directory_file::close()
{
   if( m_dir != 0 )
   {
      if( ::closedir( m_dir ) != 0 )
      {
         throw new IOError(ErrorParam(e_io_close, __LINE__,__FILE__ )
            .extra("::closedir")
            .sysError(errno) );
      }

      m_dir = 0;
   }
}


bool Directory_file::read( String& tgt )
{
   struct dirent *res;

   if( ::readdir_r( m_dir, &m_de, &res ) != 0 )
   {
      throw new IOError(ErrorParam(e_io_close, __LINE__,__FILE__ )
         .extra("::closedir")
         .sysError(errno) );
   }

   if( res == 0 )
   {
      return false;
   }

   tgt.fromUTF8( res->d_name );
   return true;
}

//========================================================================
//

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
      FStream* fs = new FStream( &handle );
      return fs;
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
         FStream* fs = new FStream( &handle );
         return fs;
      }
      else
         return 0;
   }

   bSuccess = false;
   return 0;
}


Directory* VFSFile::openDir( const URI& uri )
{
   AutoCString filename( uri.path() );

   DIR *dir = ::opendir( filename.c_str() );
   if ( dir == 0 ) {
      return 0;
   }

   return new Directory_file( uri.path(), dir );
}


bool VFSFile::readStats( const URI& uri, FileStat &sts )
{
   AutoCString filename( uri.get() );

   struct stat fs;

   if ( lstat( filename.c_str(), &fs ) != 0 )
   {
      return false;
   }

   sts.size(fs.st_size);
   if( S_ISREG( fs.st_mode ) )
      sts.type(FileStat::_normal);
   else if( S_ISDIR( fs.st_mode ) )
      sts.type(FileStat::_dir);
   else if( S_ISFIFO( fs.st_mode ) )
      sts.type(FileStat::_pipe);
   else if( S_ISLNK( fs.st_mode ) )
      sts.type(FileStat::_link);
   else if( S_ISBLK( fs.st_mode ) || S_ISCHR( fs.st_mode ) )
      sts.type(FileStat::_device);
   else if( S_ISSOCK( fs.st_mode ) )
      sts.type(FileStat::_socket);
   else
      sts.type(FileStat::_unknown);

   /*
   sts.m_access = fs.st_mode;
   sts.m_owner = fs.st_uid;      
   sts.m_group = fs.st_gid;
   */

   sts.atime().fromSystemTime( &fs.st_atime );    /* time of last access */
   sts.ctime().fromSystemTime( &fs.st_ctime );     /* time of last change */

   // copy last change time to last modify time
   sts.mtime().fromSystemTime( &fs.st_ctime );     /* time of last change */

   return true;
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
