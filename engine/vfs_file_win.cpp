/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_file_win.cpp

   VSF provider for physical file system on the host system MS-Windows
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
#include <falcon/autowstring.h>
#include <falcon/fstream.h>
#include <falcon/fstream_win.h>
#include <falcon/errors/ioerror.h>
#include <falcon/filestat.h>
#include <falcon/directory.h>

#include <windows.h>

namespace Falcon {

class FALCON_DYN_CLASS Directory_file: public Directory
{
public:
   Directory_file( const URI& location, HANDLE md, WIN32_FIND_DATAW fd );
   virtual ~Directory_file();
   virtual void close();
   virtual bool read( String& tgt );
   
private:
   HANDLE m_handle;
   WIN32_FIND_DATAW m_raw_dir;
   bool m_first;
};


Directory_file::Directory_file( const URI& location, HANDLE md, WIN32_FIND_DATAW fd ):
   Directory(location),
   m_handle(md),
   m_raw_dir( fd ),
   m_first( true )
{     
}

Directory_file::~Directory_file()
{
   close();
}


void Directory_file::close()
{
   if ( m_handle != INVALID_HANDLE_VALUE ) {
      if ( ! FindClose( m_handle ) ) {
         throw new IOError( ErrorParam( e_io_close, __LINE__, __FILE__ )
            .extra( "Directory::close" )
            .sysError( GetLastError() ) );
      }
   }

   m_handle = INVALID_HANDLE_VALUE;
}


bool Directory_file::read( String& str )
{
   if( m_handle == INVALID_HANDLE_VALUE )
      return false;

	bool bWideChar = true;

   if ( m_first ) {
      m_first = false;
   }
   else 
   {
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

//========================================================================
//


static DWORD win_paramsToMode( VFSFile::OParams p )
{
   DWORD omode;

   if ( p.isRdwr() )
      omode = GENERIC_READ | GENERIC_WRITE;
   else if ( p.isRdOnly() )
      omode = GENERIC_READ;
   else
      omode = GENERIC_WRITE;

   return omode;
}


static DWORD win_paramsToShare( VFSFile::OParams p )
{
   DWORD shmode;
   if ( p.isShNone() )
      shmode = 0;
   else if( p.isShNoRead() )
      shmode = FILE_SHARE_WRITE;
   else if ( p.isShNoWrite() )
		shmode = FILE_SHARE_READ;
	else
		shmode = FILE_SHARE_READ | FILE_SHARE_WRITE;
   
   return shmode;
}


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
   DWORD omode = win_paramsToMode( p );
   DWORD oshare = win_paramsToShare( p );

   String path = uri.path();
   Path::uriToWin( path );
   AutoWString wstr( path );

   HANDLE handle = CreateFileW( wstr.w_str(),
      omode,
      oshare,
      NULL,
      OPEN_EXISTING,
      0,
      NULL );

	DWORD dwError = GetLastError();
   if ( handle == 0 || handle == INVALID_HANDLE_VALUE )
   {
      if ( dwError  == ERROR_CALL_NOT_IMPLEMENTED )
      {
         AutoCString cstr( path );
         handle = CreateFile( cstr.c_str(),
               omode,
               oshare,
               NULL,
               OPEN_EXISTING,
               0,
               NULL );
      }
   }

   if ( handle == 0 || handle == INVALID_HANDLE_VALUE )
   {
      throw new IOError( ErrorParam( e_io_open, __LINE__, __FILE__ )
         .extra( path )
         .sysError( GetLastError() ) );
   }

   return new FStream( new WinFStreamData(handle) );
}


Stream *VFSFile::create( const URI& uri, const CParams &p )
{
   DWORD omode = win_paramsToMode( p );
   DWORD oshare = win_paramsToShare( p );
   DWORD ocreate = p.isNoOvr() ? 0 : CREATE_ALWAYS;
   
   // turn the xxx bytes 
   DWORD oattribs = FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_ARCHIVE;

   String path = uri.path();
   Path::uriToWin( path );
   AutoWString wstr( path );

   HANDLE handle = CreateFileW( wstr.w_str(),
      omode,
      oshare,
      NULL,
      ocreate,
      oattribs,
      NULL );

	DWORD dwError = GetLastError();
   if ( handle == 0 || handle == INVALID_HANDLE_VALUE )
   {
      if ( dwError == ERROR_CALL_NOT_IMPLEMENTED )
      {
         AutoCString cstr( path );
         handle = CreateFile( cstr.c_str(),
               omode,
               oshare,
               NULL,
               ocreate,
               oattribs,
               NULL );
      }
      dwError = GetLastError();
   }

   if ( handle == 0 || handle == INVALID_HANDLE_VALUE )
   {
      throw new IOError( ErrorParam( e_io_creat, __LINE__, __FILE__ )
         .extra( path )
         .sysError( dwError ) );
   }

   // the caller may not really want to open the stream.
   if( p.isNoStream() )
   {
      CloseHandle( handle );
      return 0;
   }
      
   return new FStream( new WinFStreamData(handle) );
}


Directory* VFSFile::openDir( const URI& uri )
{
   String fname = uri.path() + "\\*";
   Path::uriToWin( fname );
   AutoWString wBuffer( fname );

   WIN32_FIND_DATAW dir_data;
   HANDLE handle = FindFirstFileW( wBuffer.w_str(), &dir_data );
	if( handle == INVALID_HANDLE_VALUE && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		AutoCString cBuffer( fname );
      handle = FindFirstFile( cBuffer.c_str(), (WIN32_FIND_DATA*) &dir_data );
	}

   if ( handle == INVALID_HANDLE_VALUE )
   {
      throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
         .extra(fname)
         .sysError( GetLastError() ) );
   }
      
   return new Directory_file( uri.path(), handle, dir_data );
}


FileStat::t_fileType VFSFile::fileType( const URI& uri, bool )
{
   String fname = uri.path();
   Path::uriToWin( fname );

	AutoWString wBuffer( fname );
   // First, determine if the file exists
   if( fname.size() > 0 && fname.getCharAt(fname.length()-1) != '.' )
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
               return FileStat::_notFound;

            FindClose( hFound );

            // check case sensitive
            String ffound(aFindData.cFileName);
            if( fname.subString( fname.length() - ffound.length() ) != ffound )
               return FileStat::_notFound;
         }
         else
            return FileStat::_notFound;
      }
      
      FindClose( hFound );
      
      // Then, see if the case matches.
      String ffound(wFindData.cFileName);
      if( fname.subString( fname.length() - ffound.length() ) != ffound )
         return FileStat::_notFound;
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

      CloseHandle( temp );

      if( attribs == INVALID_FILE_ATTRIBUTES ) {
         return FileStat::_unknown;
      }

      if( (attribs & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY ) {
         return FileStat::_dir;
      }
   }

   BY_HANDLE_FILE_INFORMATION info;
   memset( &info, 0, sizeof( info ) );

   GetFileInformationByHandle( temp, &info );
   CloseHandle( temp );

   if( info.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY )
      return FileStat::_dir;
   else
      return FileStat::_normal;
}


bool VFSFile::readStats( const URI& uri, FileStat &sts, bool )
{
   String fname = uri.path();
   Path::uriToWin( fname );

	AutoWString wBuffer( fname );
   // First, determine if the file exists
   if( fname.size() > 0 && fname.getCharAt(fname.length()-1) != '.' )
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

      CloseHandle( temp );

      if( attribs == INVALID_FILE_ATTRIBUTES ) {
         return false;
      }

      if( (attribs & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY ) {
         sts.type(FileStat::_dir);
         sts.size(0);
         return true;
      }
      return false;
   }

   BY_HANDLE_FILE_INFORMATION info;
   memset( &info, 0, sizeof( info ) );

   GetFileInformationByHandle( temp, &info );

   if( info.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY )
      sts.type(FileStat::_dir);
   else
      sts.type(FileStat::_normal);

   FILETIME local_timing;
   SYSTEMTIME timing;

   FileTimeToLocalFileTime( &info.ftCreationTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &timing );
   sts.ctime().fromSystemTime( &timing );

   FileTimeToLocalFileTime( &info.ftLastAccessTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &timing );
   sts.atime().fromSystemTime( &timing );

   FileTimeToLocalFileTime( &info.ftLastWriteTime, &local_timing );
   FileTimeToSystemTime( &local_timing, &timing );
   sts.mtime().fromSystemTime( &timing );

   int64 size = info.nFileSizeHigh;
   size <<= 32;
   size |= info.nFileSizeLow;
   sts.size(size);
   
   CloseHandle( temp );

   return true;
}


void VFSFile::erase( const URI &uri )
{
   String fname = uri.path();
   Path::uriToWin( fname );

   AutoWString wBuffer( fname );
   BOOL res = DeleteFileW( wBuffer.w_str() );
	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
      AutoCString cBuffer( fname );
      res = DeleteFile( cBuffer.c_str() );
	}

   if ( ! res ) {
      throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
         .extra( fname ) 
         .sysError(::GetLastError()));
   }
}


void VFSFile::move( const URI &suri, const URI &duri )
{
   String srcPath = suri.path();
   Path::uriToWin( srcPath );
   String destPath = duri.path();
   Path::uriToWin( destPath );

   BOOL res = ::MoveFileW(AutoWString(srcPath).w_str(), AutoWString( destPath ).w_str() );   
   if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
   {
      res = ::MoveFileA(AutoCString(srcPath).c_str(), AutoCString( destPath ).c_str() );
   }

   if ( ! res ) 
   {
      throw new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
         .extra( "move" ) 
         .sysError(::GetLastError()));
   }
}


static void _mkdir( const String& fname )
{
   AutoWString wBuffer( fname );
   BOOL res = ::CreateDirectoryW( wBuffer.w_str(), NULL );

	if( ! res && GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
	{
		AutoCString cBuffer( fname );
      res = CreateDirectory( cBuffer.c_str(), NULL );
	}

   if ( ! res ) 
   {
      new IOError( ErrorParam( e_io_error, __LINE__, __FILE__ )
         .extra( fname )
         .sysError( GetLastError() ) );
   }
}

void VFSFile::mkdir( const URI &uri, bool descend )
{
   String strName = uri.path();
   Path::uriToWin( strName );

   if ( descend )
   {
      // find /.. sequences
      uint32 pos = strName.find( "\\" );
      if(pos == 0) pos = strName.find( "\\", 1 ); // an absolute path
      while( true )
      {
         String strPath( strName, 0, pos );

         // stat the file
         FileStat fstats;

         // if the file exists...
         if ( fileType( strPath ) != FileStat::_dir )
         {
            // if it's not a directory, try to create the directory.
            _mkdir( strPath );
         }

         // last loop?
         if ( pos == String::npos )
            break;

         pos = strName.find( "\\", pos + 1 );
       }

   }
   else
   {
      // Just one try; succeed or fail
      _mkdir( strName );
   }
}

}

/* end of vsf_file_windows.cpp */
