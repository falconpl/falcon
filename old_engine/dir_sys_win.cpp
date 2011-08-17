/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: dir_win.cpp

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

#include <falcon/autowstring.h>
#include <falcon/autocstring.h>
#include <falcon/dir_sys_win.h>
#include <falcon/time_sys_win.h>
#include <falcon/timestamp.h>
#include <falcon/path.h>

#include <cstring>

#include <falcon/item.h>
#include <falcon/mempool.h>
#include <falcon/memory.h>
#include <falcon/string.h>
#include <falcon/sys.h>

namespace Falcon {
namespace Sys {

bool fal_fileType( const String &filename, FileStat::e_fileType &st )
{
   String fname = filename;
   Path::uriToWin( fname );

   AutoWString wstrBufName( fname );
	DWORD attribs = GetFileAttributesW( wstrBufName.w_str() );

	if( attribs == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		AutoCString cstrBufName( fname );
		attribs = GetFileAttributes( cstrBufName.c_str() );
	}

   if( attribs == INVALID_FILE_ATTRIBUTES ) {
      return false;
   }

   if( (attribs & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY )
      st = FileStat::t_dir;
   else
      st = FileStat::t_normal;

   return true;
}

bool fal_stats( const String &filename, FileStat &sts )
{
   String fname = filename;
   Path::uriToWin( fname );

	AutoWString wBuffer( fname );
   // First, determine if the file exists
   if( filename.size() > 0 && filename.getCharAt(filename.length()-1) != '.' )
   {
      WIN32_FIND_DATAW wFindData;

      HANDLE hFound = FindFirstFileW( wBuffer.w_str(), &wFindData );
      if( hFound == INVALID_HANDLE_VALUE )
      {
         if( GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
         {
            WIN32_FIND_DATAA aFindData;
            AutoCString cBuffer( fname );
            
            hFound = FindFirstFileA( cBuffer.c_str(), &aFindData );
            
            if ( hFound == INVALID_HANDLE_VALUE )
               return false;

            FindClose( hFound );

            // check case sensitive
            String ffound(aFindData.cFileName);
            if( fname.subString( fname.length() - ffound.length() ) != ffound )
               return false;
         }
         else
            return false;
      }
      
      FindClose( hFound );
      
      // Then, see if the case matches.
      String ffound(wFindData.cFileName);
      if( fname.subString( fname.length() - ffound.length() ) != ffound )
         return false;
   }

   // ok, file exists and with matching case

   HANDLE temp = CreateFileW( wBuffer.w_str(),
      GENERIC_READ,
      FILE_SHARE_READ,
      NULL,
      OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS,
      NULL );

	if( (temp == INVALID_HANDLE_VALUE || temp == 0) && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
      AutoCString cBuffer( fname );
      temp = CreateFile( cBuffer.c_str(),
	   		GENERIC_READ,
				FILE_SHARE_READ,
				NULL,
				OPEN_EXISTING,
				FILE_FLAG_BACKUP_SEMANTICS,
				NULL );
	}

   if( temp == INVALID_HANDLE_VALUE ) 
   {
      // on win 95/98, we can't normally access directory data.
		
      DWORD attribs = GetFileAttributesW( wBuffer.w_str() );
		if(  attribs == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
		{
         AutoCString cBuffer( fname );
			attribs = GetFileAttributes( cBuffer.c_str() );
		}

      if( attribs == INVALID_FILE_ATTRIBUTES ) {
         return false;
      }

      if( (attribs & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY ) {
         sts.m_type = FileStat::t_dir;
         sts.m_attribs = attribs;
         sts.m_size = 0;
         sts.m_mtime = new TimeStamp();
         sts.m_atime = new TimeStamp();
         sts.m_ctime = new TimeStamp();
         sts.m_owner = 0;      /* user ID of owner */
         sts.m_group = 0;      /* group ID of owner */
         return true;
      }
      return false;
   }

   BY_HANDLE_FILE_INFORMATION info;
   memset( &info, 0, sizeof( info ) );

   GetFileInformationByHandle( temp, &info );

   if( info.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY )
      sts.m_type = FileStat::t_dir;
   else
      sts.m_type = FileStat::t_normal;

   FILETIME local_timing;
   SYSTEMTIME timing;

   FileTimeToLocalFileTime( &info.ftCreationTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &timing );
   WinSystemTime mtime( timing );
   if ( sts.m_ctime == 0 )
      sts.m_ctime = new TimeStamp();
   sts.m_ctime->fromSystemTime( mtime );

   FileTimeToLocalFileTime( &info.ftLastAccessTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &mtime.m_time );
   if ( sts.m_atime == 0 )
      sts.m_atime = new TimeStamp();
   sts.m_atime->fromSystemTime( mtime );

   FileTimeToLocalFileTime( &info.ftLastWriteTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &mtime.m_time );
   if ( sts.m_mtime == 0 )
      sts.m_mtime = new TimeStamp();
   sts.m_mtime->fromSystemTime( mtime );

   sts.m_size = info.nFileSizeHigh;
   sts.m_size = sts.m_size << 32 | info.nFileSizeLow;
   sts.m_attribs = info.dwFileAttributes;
   sts.m_owner = 0;      /* user ID of owner */
   sts.m_group = 0;      /* group ID of owner */

   CloseHandle( temp );

   return true;
}

bool fal_mkdir( const String &filename, int32 &fsStatus )
{
   String fname = filename;
   Path::uriToWin( fname );

   AutoWString wBuffer( fname );
   BOOL res = CreateDirectoryW( wBuffer.w_str(), NULL );

	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		AutoCString cBuffer( fname );
      res = CreateDirectory( cBuffer.c_str(), NULL );
	}

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}


bool fal_mkdir( const String &strName, int32 &fsError, bool descend )
{
   if ( descend )
   {
      // find /.. sequences
      uint32 pos = strName.find( "/" );
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

bool fal_rmdir( const String &filename, int32 &fsStatus )
{
   String fname = filename;
   Path::uriToWin( fname );

   AutoWString wBuffer( fname );
   BOOL res = RemoveDirectoryW( wBuffer.w_str() );
	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
      AutoCString cBuffer( fname );
		res = RemoveDirectory( cBuffer.c_str() );
	}

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_unlink( const String &filename, int32 &fsStatus )
{
   String fname = filename;
   Path::uriToWin( fname );

   AutoWString wBuffer( fname );
   BOOL res = DeleteFileW( wBuffer.w_str() );
	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
      AutoCString cBuffer( fname );
      res = DeleteFile( cBuffer.c_str() );
	}

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_move( const String &filename, const String &dest, int32 &fsStatus )
{
   String fname1 = filename;
   Path::uriToWin( fname1 );
   String fname2 = dest;
   Path::uriToWin( fname2 );

   AutoWString wBuffer1( fname1 );
   AutoWString wBuffer2( fname2 );
   BOOL res = MoveFileW( wBuffer1.w_str(), wBuffer2.w_str() );

	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
      AutoCString cBuffer1( fname1 );
      AutoCString cBuffer2( fname2 );
      res = MoveFile( cBuffer1.c_str(), cBuffer2.c_str() );
	}

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_chdir( const String &filename, int32 &fsStatus )
{
   String fname = filename;
   Path::uriToWin( fname );
   
   AutoWString wBuffer( fname );

   BOOL res = SetCurrentDirectoryW( wBuffer.w_str() );
	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
      AutoCString cBuffer( fname );
      res = SetCurrentDirectory( cBuffer.c_str() );
	}

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_getcwd( String &cwd, int32 &fsError )
{
   DWORD size = GetCurrentDirectory( 0, NULL );
   if( size == 0 ) {
      fsError = GetLastError();
      return 0;
   }

	int bufSize = size * sizeof( wchar_t ) + sizeof( wchar_t );
   wchar_t *buffer = (wchar_t *) memAlloc( bufSize );
   size = GetCurrentDirectoryW( bufSize, buffer );
	if( size == 0 && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *buffer_c = (char *) buffer;
		size = GetCurrentDirectory( bufSize, buffer_c );

		if( size == 0 ) {
			memFree( buffer );
			fsError = GetLastError();
			return false;
		}

		cwd.adopt( buffer_c, size, bufSize );
		Path::winToUri( cwd );
		return true;
	}

   if( size == 0 ) {
      memFree( buffer );
      fsError = GetLastError();
      return false;
   }

   cwd.adopt( buffer, size, bufSize );
   Path::winToUri( cwd );

   return true;
}


bool fal_chmod( const String &fname, uint32 mode )
{
   return false;
}

bool fal_chown( const String &fname, int32 owner )
{
   return false;
}

bool fal_chgrp( const String &fname, int32 owner )
{
   return false;
}

bool fal_readlink( const String &fname, String &link )
{
   /** TODO: implement on windows */
   return false;
}

bool fal_writelink( const String &fname, String &link )
{
   /** TODO: implement on windows */
   return false;
}

::Falcon::DirEntry *fal_openDir( const String &path, int32 &fsError )
{
   String fname = path + "\\*";
   Path::uriToWin( fname );

   AutoWString wBuffer( fname );

   WIN32_FIND_DATAW dir_data;
   HANDLE handle = FindFirstFileW( wBuffer.w_str(), &dir_data );
	if( handle == INVALID_HANDLE_VALUE && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		AutoCString cBuffer( fname );
      handle = FindFirstFile( cBuffer.c_str(), (WIN32_FIND_DATA*) &dir_data );
	}

   if ( handle != INVALID_HANDLE_VALUE )
      return new DirEntry_win( path, handle, dir_data );

   fsError = GetLastError();
   return 0;
}

void fal_closeDir( ::Falcon::DirEntry *entry )
{
	delete entry;
}

} // Namespace Srv

bool DirEntry_win::read( String &str )
{
   if( m_handle == INVALID_HANDLE_VALUE )
      return 0;

	bool bWideChar = true;

   if ( m_first ) {
      m_first = false;
   }
   else {
      if ( ! FindNextFileW( m_handle, &m_raw_dir ) )
		{
			if( GetLastError() != ERROR_CALL_NOT_IMPLEMENTED )
				return false;

			bWideChar = false;
			if ( ! FindNextFile( m_handle, (WIN32_FIND_DATA*) &m_raw_dir ) )
				return false;
		}
   }

	if( bWideChar )
		str.bufferize( m_raw_dir.cFileName );
	else
		str.bufferize( ((WIN32_FIND_DATA*) &m_raw_dir)->cFileName );

   Path::winToUri( str );
   return true;
}

void DirEntry_win::close()
{
   if ( m_handle != INVALID_HANDLE_VALUE ) {
      if ( ! FindClose( m_handle ) ) {
         m_lastError = GetLastError();
      }
      else
         m_lastError = 0;
   }
   m_handle = INVALID_HANDLE_VALUE;
}

}


/* end of dir_win.cpp */
