/*
   FALCON - The Falcon Programming Language.
   FILE: dir_unix.cpp

   Implementation of directory system support for unix.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of directory system support for unix.
*/

#include <falcon/item.h>
#include <falcon/mempool.h>
#include <falcon/string.h>
#include <falcon/memory.h>
#include <falcon/autocstring.h>

#include <falcon/dir_sys_unix.h>
#include <falcon/time_sys_unix.h>
#include <falcon/timestamp.h>
#include <falcon/filestat.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <stdio.h>


namespace Falcon {
namespace Sys {

bool fal_fileType( const String &fname, FileStat::e_fileType &st )
{
   AutoCString filename(fname );
   struct stat fs;

   if ( lstat( filename.c_str(), &fs ) != 0 ) {
      st = FileStat::t_notFound;
      return false;
   }

   if( S_ISREG( fs.st_mode ) )
      st = FileStat::t_normal;
   else if( S_ISDIR( fs.st_mode ) )
      st = FileStat::t_dir;
   else if( S_ISFIFO( fs.st_mode ) )
      st = FileStat::t_pipe;
   else if( S_ISLNK( fs.st_mode ) )
      st = FileStat::t_link;
   else if( S_ISBLK( fs.st_mode ) || S_ISCHR( fs.st_mode ) )
      st = FileStat::t_device;
   else if( S_ISSOCK( fs.st_mode ) )
      st = FileStat::t_socket;
   else
      st = FileStat::t_unknown;

   return true;
}

bool fal_stats( const String &f, FileStat &sts )
{
   AutoCString filename( f );

   struct stat fs;

   if ( lstat( filename.c_str(), &fs ) != 0 )
   {
      return false;
   }

   sts.m_size = fs.st_size;
   if( S_ISREG( fs.st_mode ) )
      sts.m_type = FileStat::t_normal;
   else if( S_ISDIR( fs.st_mode ) )
      sts.m_type = FileStat::t_dir;
   else if( S_ISFIFO( fs.st_mode ) )
      sts.m_type = FileStat::t_pipe;
   else if( S_ISLNK( fs.st_mode ) )
      sts.m_type = FileStat::t_link;
   else if( S_ISBLK( fs.st_mode ) || S_ISCHR( fs.st_mode ) )
      sts.m_type = FileStat::t_device;
   else if( S_ISSOCK( fs.st_mode ) )
      sts.m_type = FileStat::t_socket;
   else
      sts.m_type = FileStat::t_unknown;

   sts.m_access = fs.st_mode;
   sts.m_owner = fs.st_uid;      /* user ID of owner */
   sts.m_group = fs.st_gid;      /* group ID of owner */


   UnixSystemTime mtime( fs.st_mtime );

   if (sts.m_atime == 0 )
      sts.m_atime = new TimeStamp();
   mtime.m_time_t = fs.st_atime;
   sts.m_atime->fromSystemTime( mtime );    /* time of last access */

   if (sts.m_ctime == 0 )
      sts.m_ctime = new TimeStamp();
   mtime.m_time_t = fs.st_ctime;
   sts.m_ctime->fromSystemTime( mtime );     /* time of last change */

   // copy last change time to last modify time
   if (sts.m_mtime == 0 )
      sts.m_mtime = new TimeStamp();
   sts.m_mtime->fromSystemTime( mtime );

   return true;
}

bool fal_mkdir( const String &f, int32 &fsStatus )
{
   AutoCString filename( f );

   if ( ::mkdir( filename.c_str(), 0744 ) == 0 ) {
      fsStatus = 0;
      return true;
   }
   fsStatus = errno;
   return false;
}

bool fal_mkdir( const String &strName, int32 &fsError, bool descend )
{
   if ( descend )
   {
      // find /.. sequences
      uint32 pos = strName.find( "/" );
      if(pos == 0) pos = strName.find( "/", 1 ); // an absolute path
      while( true )
      {
         String strPath( strName, 0, pos );

         // stat the file
         FileStat fstats;
         // if the file exists...
         if ( (! Sys::fal_stats( strPath, fstats )) ||
              fstats.m_type != FileStat::t_dir )
         {
            // if it's not a directory, try to create the directory.
            if ( ! Sys::fal_mkdir( strPath, fsError ) )
               return false;
         }

         // last loop?
         if ( pos == String::npos )
            break;

         pos = strName.find( "/", pos + 1 );
       }

   }
   else
   {
      // Just one try; succeed or fail
      return Sys::fal_mkdir( strName, fsError );
   }

   return true;
}

bool fal_rmdir( const String &f, int32 &fsStatus )
{
   AutoCString filename( f );

   if ( ::rmdir( filename.c_str() ) == 0 ) {
      fsStatus = 0;
      return true;
   }
   fsStatus = errno;
   return false;
}

bool fal_unlink( const String &f, int32 &fsStatus )
{
   AutoCString filename( f );

   if ( ::unlink( filename.c_str() ) == 0 ) {
      fsStatus = 0;
      return true;
   }

   fsStatus = errno;
   return false;
}

bool fal_move( const String &f, const String &d, int32 &fsStatus )
{
   AutoCString filename( f );
   AutoCString dest( d );

   if ( ::rename( filename.c_str(), dest.c_str() ) == 0 )
   {
      fsStatus = 0;
      return true;
   }
   fsStatus = errno;
   return false;
}

bool fal_chdir( const String &f, int32 &fsStatus )
{
   AutoCString filename( f );

   if ( ::chdir( filename.c_str() ) == 0 ) {
      fsStatus = 0;
      return true;
   }
   fsStatus = errno;
   return false;
}

bool fal_getcwd( String &fname, int32 &fsError )
{

   char buf[256];
   uint32 pwdSize = 256;
   char *buffer = buf;
   char *bufret;

   while ( (bufret = ::getcwd( buffer, pwdSize )) == 0 && errno == ERANGE ) {
      pwdSize += 256;
      if ( buffer != buf )
         memFree( buffer );
      buffer = ( char * ) memAlloc( pwdSize );
   }

   fsError = errno;
   bool val;
   if ( bufret != 0 )
   {
      val = true;
      fname.fromUTF8( bufret );
   }
   else
      val = false;

   if ( buffer != buf )
      memFree( buffer );

   return val;
}

bool fal_chmod( const String &fname, uint32 mode )
{
   AutoCString filename( fname );
   bool ret = ::chmod( filename.c_str(), mode ) == 0;
   return ret;
}

bool fal_chown( const String &fname, int32 owner )
{
   AutoCString filename( fname );
   bool ret = ::chown( filename.c_str(), owner , (gid_t) -1 ) == 0;
   return ret;
}

bool fal_readlink( const String &fname, String &link )
{
   char buf[1024];
   int len;
   AutoCString filename( fname );

   if ( ( len = readlink( filename.c_str(), buf, sizeof(buf) - 1 ) ) != -1) {
      buf[len] = '\0';
      link.fromUTF8( buf );
      return true;
   }
   return false;
}

bool fal_writelink( const String &fname, const String &link )
{
   AutoCString filename( fname );
   AutoCString linkname( link );

   if ( ! symlink( filename.c_str(), linkname.c_str() ) )
   {
      return false;
   }

   return true;
}

bool fal_chgrp( const String &fname, int32 owner )
{
   AutoCString filename( fname );
   bool ret = ::chown( filename.c_str(), (uid_t) -1, owner ) == 0;
   return ret;
}

::Falcon::DirEntry *fal_openDir( const String &p, int32 &fsStatus )
{
   AutoCString filename( p );

   DIR *dir = ::opendir( filename.c_str() );
   if ( dir == 0 ) {
      fsStatus = errno;
      return 0;
   }

   return new DirEntry_unix( p, dir );
}

void fal_closeDir( ::Falcon::DirEntry *entry )
{
	delete entry;
}

} // SYS namespace



bool DirEntry_unix::read( String &res )
{
   // Glibc doesn't perform that check
   if( m_raw_dir == 0)
   {
      return false;
   }
   struct dirent *d;

   errno = 0;
   d = readdir( m_raw_dir );
   m_lastError = errno;

   if ( d == 0 ) {
      return false;
   }

   res.fromUTF8( d->d_name );
   return true;
}

void DirEntry_unix::close()
{
   if ( m_raw_dir != 0 ) {
      errno = 0;
      closedir( m_raw_dir );
      m_lastError = errno;
   }
   m_raw_dir = 0;
}


}


/* end of dir_unix.cpp */
