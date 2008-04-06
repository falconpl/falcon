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

#include <falcon/dir_sys_win.h>
#include <falcon/time_sys_win.h>
#include <falcon/timestamp.h>

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
   Sys::falconToWin_fname( fname );

	uint32 bufSize = fname.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
	fname.toWideString( bufname, bufSize );

	DWORD attribs = GetFileAttributesW( bufname );

	if( attribs == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) bufname;
		if( fname.toCString( bufname_c, bufSize ) > 0 )
			attribs = GetFileAttributes( bufname_c );
	}

	memFree( bufname );

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
   Sys::falconToWin_fname( fname );

	uint32 bufSize = fname.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
	fname.toWideString( bufname, bufSize );

   HANDLE temp = CreateFileW( bufname,
      GENERIC_READ,
      FILE_SHARE_READ,
      NULL,
      OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS,
      NULL );

	if( (temp == INVALID_HANDLE_VALUE || temp == 0) && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) bufname;
		if( fname.toCString( bufname_c, bufSize ) > 0 )
			temp = CreateFile( bufname_c,
				GENERIC_READ,
				FILE_SHARE_READ,
				NULL,
				OPEN_EXISTING,
				FILE_FLAG_BACKUP_SEMANTICS,
				NULL );
	}

   if( temp == INVALID_HANDLE_VALUE ) {
      // on win 95/98, we can't normally access directory data.
		uint32 bufSize = (fname.length()+1) * sizeof( wchar_t );
		wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
		filename.toWideString( bufname, bufSize );


      DWORD attribs = GetFileAttributesW( bufname );
		if(  attribs == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
		{
			char *bufname_c = (char *) bufname;
			if( fname.toCString( bufname_c, bufSize ) > 0 )
				attribs = GetFileAttributes( bufname_c );
		}

		memFree( bufname );

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

	memFree( bufname );

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
   sts.m_ctime = new TimeStamp();
   sts.m_ctime->fromSystemTime( mtime );

   FileTimeToLocalFileTime( &info.ftLastAccessTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &mtime.m_time );
   sts.m_atime = new TimeStamp();
   sts.m_atime->fromSystemTime( mtime );

   FileTimeToLocalFileTime( &info.ftLastWriteTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &mtime.m_time );
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
   Sys::falconToWin_fname( fname );

	uint32 bufSize = fname.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
	fname.toWideString( bufname, bufSize );

   BOOL res = CreateDirectoryW( bufname, NULL );

	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) bufname;
		if( fname.toCString( bufname_c, bufSize ) > 0 )
			res = CreateDirectory( bufname_c, NULL );
	}

	memFree( bufname );

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_rmdir( const String &filename, int32 &fsStatus )
{
   String fname = filename;
   Sys::falconToWin_fname( fname );

	uint32 bufSize = fname.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
	fname.toWideString( bufname, bufSize );

   BOOL res = RemoveDirectoryW( bufname );
	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) bufname;
		if( fname.toCString( bufname_c, bufSize ) > 0 )
			res = RemoveDirectory( bufname_c );
	}

	memFree( bufname );

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_unlink( const String &filename, int32 &fsStatus )
{
   String fname = filename;
   Sys::falconToWin_fname( fname );

	uint32 bufSize = fname.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
	fname.toWideString( bufname, bufSize );

   BOOL res = DeleteFileW( bufname );
	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) bufname;
		if( fname.toCString( bufname_c, bufSize ) > 0 )
			res = DeleteFile( bufname_c );
	}

	memFree( bufname );

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_move( const String &filename, const String &dest, int32 &fsStatus )
{
   String fname1 = filename;
   Sys::falconToWin_fname( fname1 );
   String fname2 = dest;
   Sys::falconToWin_fname( fname2 );

	uint32 bufSize1 = fname1.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname1 = (wchar_t *) memAlloc( bufSize1 );
	fname1.toWideString( bufname1, bufSize1 );

	uint32 bufSize2 = fname2.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname2 = (wchar_t *) memAlloc( bufSize2 );
	fname2.toWideString( bufname2, bufSize2 );

   BOOL res = MoveFileW( bufname1, bufname2 );

	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c1 = (char *) bufname1;
		if( fname1.toCString( bufname_c1, bufSize1 ) > 0 )
		{
			char *bufname_c2 = (char *) bufname2;
			if( fname2.toCString( bufname_c2, bufSize2 ) > 0 )
				res = MoveFile( bufname_c1, bufname_c2 );
		}
	}


	memFree( bufname1 );
	memFree( bufname2 );

   if ( res == TRUE ) {
      return true;
   }
   fsStatus = GetLastError();
   return false;
}

bool fal_chdir( const String &filename, int32 &fsStatus )
{
   String fname = filename;
   Sys::falconToWin_fname( fname );

	uint32 bufSize = fname.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
	fname.toWideString( bufname, bufSize );

   BOOL res = SetCurrentDirectoryW( bufname );
	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) bufname;
		if( fname.toCString( bufname_c, bufSize ) > 0 )
			res = SetCurrentDirectory( bufname_c );
	}

	memFree( bufname );

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
		Sys::falconConvertWinFname( cwd );
		return true;
	}

   if( size == 0 ) {
      memFree( buffer );
      fsError = GetLastError();
      return false;
   }

   cwd.adopt( buffer, size, bufSize );
   Sys::falconConvertWinFname( cwd );

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
   String fname;
   Sys::falconToWin_fname( path, "\\*", fname );

	uint32 bufSize = fname.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *bufname = (wchar_t *) memAlloc( bufSize );
	fname.toWideString( bufname, bufSize );

   WIN32_FIND_DATAW dir_data;
   HANDLE handle = FindFirstFileW( bufname, &dir_data );
	if( handle == INVALID_HANDLE_VALUE && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		char *bufname_c = (char *) bufname;
		if( fname.toCString( bufname_c, bufSize ) > 0 )
			handle = FindFirstFile( bufname_c, (WIN32_FIND_DATA*) &dir_data );
	}

	memFree( bufname );

   if ( handle != INVALID_HANDLE_VALUE )
      return new DirEntry_win( handle, dir_data );

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

   Sys::falconConvertWinFname( str );
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
