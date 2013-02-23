/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_file_win.cpp

   VSF provider for physical file system on the host system (POSIX)
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
#include <falcon/fstream.h>
#include <falcon/errors/ioerror.h>
#include <falcon/filestat.h>
#include <falcon/directory.h>

#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <stdio.h>
#include <fcntl.h>

#define DEFAULT_CREATE_MODE 0640

namespace Falcon {

class FALCON_DYN_CLASS Directory_file: public Directory
{
public:
   Directory_file( const URI& location, DIR* md ):
      Directory(location),
      m_dir(md)
   {
      AutoCString loc(location.encode());
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
      FStream* fs = new FStream( new Sys::FileData(handle) );
      return fs;
   }

   throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
                     .extra( uri.path() )
                     .sysError(errno));
}


Stream *VFSFile::create( const URI& uri, const CParams &p )
{
   int omode = paramsToMode( p );

   if( omode == 0 )
   {
      // we must at least set WR mode.
      omode = O_WRONLY;
   }

   if ( p.isNoOvr() )
      omode |= O_EXCL;

   //TODO: something about sharing
   AutoCString cfilename( uri.path() );
   errno=0;

   int handle = ::open( cfilename.c_str(), O_CREAT | omode, DEFAULT_CREATE_MODE );

   if ( handle >= 0 )
   {
      if ( ! p.isNoStream() )
      {
         FStream* fs = new FStream( new Sys::FileData(handle) );
         return fs;
      }
      else
      {
         return 0;
      }
   }

   throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
                     .extra( uri.path() )
                     .sysError(errno));
}


Directory* VFSFile::openDir( const URI& uri )
{
   AutoCString filename( uri.path() );

   DIR *dir = ::opendir( filename.c_str() );
   if ( dir == 0 ) {
      throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
                     .sysError(errno));
   }

   return new Directory_file( uri.path(), dir );
}


FileStat::t_fileType VFSFile::fileType( const URI& uri, bool )
{
   AutoCString filename( uri.encode() );
   struct stat fs;
   if ( lstat( filename.c_str(), &fs ) != 0 )
   {
      if( errno == ENOENT )
      {
         return FileStat::_notFound;
      }

      throw new IOError( ErrorParam( e_io_creat, __LINE__, __FILE__ ).sysError( errno ));
   }

   if( S_ISREG( fs.st_mode ) )
      return FileStat::_normal;
   else if( S_ISDIR( fs.st_mode ) )
      return FileStat::_dir;
   else if( S_ISFIFO( fs.st_mode ) )
      return FileStat::_pipe;
   else if( S_ISLNK( fs.st_mode ) )
      return FileStat::_link;
   else if( S_ISBLK( fs.st_mode ) || S_ISCHR( fs.st_mode ) )
      return FileStat::_device;
   else if( S_ISSOCK( fs.st_mode ) )
      return FileStat::_socket;


   return FileStat::_unknown;
}

bool VFSFile::readStats( const URI& uri, FileStat &sts, bool )
{
   AutoCString filename( uri.encode() );

   struct stat fs;

   if ( lstat( filename.c_str(), &fs ) != 0 )
   {
      if( errno == ENOENT )
      {
         return false;
      }

      throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
                     .sysError(errno));
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

void VFSFile::erase( const URI &uri )
{
   AutoCString filename( uri.path() );
   if( ::unlink( filename.c_str() ) != 0 )
   {
      // try with rmdir
      if( ::rmdir( filename.c_str() ) != 0 )
      {
         throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
                     .sysError(errno));
      }
   }
}

void VFSFile::move( const URI &suri, const URI &duri )
{
   AutoCString filename( suri.path() );
   AutoCString dest( duri.path() );
   if( ::rename( filename.c_str(), dest.c_str() ) != 0 )
   {
      throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
                     .sysError(errno));
   }
}


void VFSFile::mkdir( const URI &uri, bool descend )
{
   String strName = uri.path();

   if ( descend )
   {
      // find /.. sequences
      uint32 pos = strName.find( '/' );
      if(pos == 0) pos = strName.find( '/', 1 ); // an absolute path
      while( true )
      {
         String strPath( strName, 0, pos );

         // stat the file
         FileStat fstats;

         // if the file exists...
         if ( fileType( strPath ) != FileStat::_dir )
         {
            // if it's not a directory, try to create the directory.
            AutoCString filename( strPath );
            if( ::mkdir( filename.c_str(), DEFAULT_CREATE_MODE ) != 0 )
            {
               throw new IOError( ErrorParam( e_io_creat, __LINE__, __FILE__ ).sysError( errno ));
            }
         }

         // last loop?
         if ( pos == String::npos )
            break;

         pos = strName.find( '/', pos + 1 );
       }

   }
   else
   {
      // Just one try; succeed or fail
      AutoCString filename( strName );
      if( ::mkdir( filename.c_str(), DEFAULT_CREATE_MODE ) != 0 )
      {
         throw new IOError( ErrorParam( e_io_creat, __LINE__, __FILE__ ).sysError( errno ));
      }
   }
}


}

/* end of vsf_file_posix.cpp */
