/*
   FALCON - The Falcon Programming Language.
   FILE: dir_unix.cpp
   $Id: dir_sys_unix.cpp,v 1.1 2007/06/21 21:54:26 jonnymind Exp $

   Implementation of directory system support for unix.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom nov 7 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Implementation of directory system support for unix.
*/

#include <falcon/item.h>
#include <falcon/mempool.h>
#include <falcon/string.h>
#include <falcon/memory.h>

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
   char *filename = (char *) memAlloc( fname.size() * 4 + 1);
   fname.toCString( filename, fname.size() * 4 );
   struct stat fs;


   if ( lstat( filename, &fs ) != 0 ) {
      st = FileStat::t_notFound;
      memFree( filename );
      return false;
   }

   memFree( filename );

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
   char *filename = (char *) memAlloc( f.size() * 4 + 1 );
   f.toCString( filename, f.size() * 4 );

   struct stat fs;

   if ( lstat( filename, &fs ) != 0 )
   {
      memFree( filename );
      return false;
   }

   memFree( filename );

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
   sts.m_mtime = new TimeStamp();
   sts.m_mtime->fromSystemTime( mtime );

   sts.m_atime = new TimeStamp();
   mtime.m_time_t = fs.st_atime;
   sts.m_atime->fromSystemTime( mtime );    /* time of last access */

   sts.m_ctime = new TimeStamp();
   mtime.m_time_t = fs.st_ctime;
   sts.m_ctime->fromSystemTime( mtime );     /* time of last change */
   return true;
}

bool fal_mkdir( const String &f, int32 &fsStatus )
{
   char *filename = (char *) memAlloc( f.size() * 4 + 1 );
   f.toCString( filename, f.size() * 4 );

   if ( ::mkdir( filename, 0777 ) == 0 ) {
      fsStatus = 0;
      memFree( filename );
      return true;
   }
   fsStatus = errno;
   memFree( filename );
   return false;
}

bool fal_rmdir( const String &f, int32 &fsStatus )
{
   char *filename = (char *) memAlloc( f.size() * 4 + 1 );
   f.toCString( filename, f.size() * 4 );

   if ( ::rmdir( filename ) == 0 ) {
      fsStatus = 0;
      memFree( filename );
      return true;
   }
   memFree( filename );
   fsStatus = errno;
   return false;
}

bool fal_unlink( const String &f, int32 &fsStatus )
{
   char *filename = (char *) memAlloc( f.size() * 4 + 1 );
   f.toCString( filename, f.size() * 4 );

   if ( ::unlink( filename ) == 0 ) {
      fsStatus = 0;
      memFree( filename );
      return true;
   }

   memFree( filename );
   fsStatus = errno;
   return false;
}

bool fal_move( const String &f, const String &d, int32 &fsStatus )
{
   char *filename = (char *) memAlloc( f.size() * 4 + 1);
   f.toCString( filename, f.size() * 4 );

   char *dest = (char *) memAlloc( d.size() * 4  + 1);
   d.toCString( filename, d.size() * 4 );

   if ( ::rename( filename, dest ) == 0 ) {
      fsStatus = 0;
      memFree( filename );
      memFree( dest );
      return true;
   }
   memFree( filename );
   memFree( dest );
   fsStatus = errno;
   return false;
}

bool fal_chdir( const String &f, int32 &fsStatus )
{
   char *filename = (char *) memAlloc( f.size() * 4 + 1 );
   f.toCString( filename, f.size() * 4 );

   if ( ::chdir( filename ) == 0 ) {
      fsStatus = 0;
      memFree( filename );
      return true;
   }
   memFree( filename );
   fsStatus = errno;
   return false;
}

bool fal_getcwd( String &fname, int32 &fsError )
{
   uint32 pwdSize = 64;

   char *buffer = (char *) memAlloc( 64 );
   char *bufret;

   while ( (bufret = ::getcwd( buffer, pwdSize )) == 0 && errno == ERANGE ) {
      memFree( buffer );
      pwdSize += 64;
      buffer = ( char * ) memAlloc( pwdSize );
   }

   if ( bufret == 0 )
   {
      memFree( buffer );
      fsError = errno;
      return 0;
   }

   fname.bufferize( buffer );
   memFree( buffer );
   return true;
}

bool fal_chmod( const String &fname, uint32 mode )
{
   char *filename = (char *) memAlloc( fname.size() * 4 + 1);
   fname.toCString( filename, fname.size() * 4 );
   bool ret = ::chmod( filename, mode ) == 0;
   memFree( filename );
}

bool fal_chown( const String &fname, int32 owner )
{
   char *filename = (char *) memAlloc( fname.size() * 4 + 1 );
   fname.toCString( filename, fname.size() * 4 );
   bool ret = ::chown( filename, owner , -1 ) == 0;
   memFree( filename );
}

bool fal_readlink( const String &fname, String &link )
{
   char buf[1024];
   int len;
   char *filename = (char *) memAlloc( fname.size() * 4 + 1 );
   fname.toCString( filename, fname.size() * 4 );

   if ( ( len = readlink( filename, buf, sizeof(buf) - 1 ) ) != -1) {
      buf[len] = '\0';
      link.bufferize( buf );
      memFree( filename );
      return true;
   }
   memFree( filename );
   return false;
}

bool fal_writelink( const String &fname, const String &link )
{
   char *filename = (char *) memAlloc( fname.size() * 4 + 1);
   fname.toCString( filename, fname.size() * 4 );
   char *linkname = (char *) memAlloc( link.size() * 4 + 1);
   link.toCString( linkname, link.size() * 4 );

   if ( ! symlink( filename, linkname ) )
   {
      memFree( filename );
      memFree( linkname );
      return false;
   }
   memFree( filename );
   memFree( linkname );

   return true;
}

bool fal_chgrp( const String &fname, int32 owner )
{
   char *filename = (char *) memAlloc( fname.size() * 4 + 1 );
   fname.toCString( filename, fname.size() * 4 );
   bool ret = ::chown( filename, -1, owner ) == 0;
   memFree( filename );
   return ret;
}

::Falcon::DirEntry *fal_openDir( const String &p, int32 &fsStatus )
{
   char *filename = (char *) memAlloc( p.size() * 4 + 1 );
   p.toCString( filename, p.size() * 4 );

   DIR *dir = ::opendir( filename );
   if ( dir == 0 ) {
      memFree( filename );
      fsStatus = errno;
      return 0;
   }
   memFree( filename );

   return new DirEntry_unix( dir );
}

void fal_closeDir( ::Falcon::DirEntry *entry )
{
	delete entry;
}

} // SYS namespace



bool DirEntry_unix::read( String &res )
{
   struct dirent *d;

   errno = 0;
   d = readdir( m_raw_dir );
   m_lastError = errno;

   if ( d == 0 ) {
      return false;
   }

   res.bufferize( d->d_name );
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
