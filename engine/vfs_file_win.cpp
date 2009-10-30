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
#include <falcon/fstream_sys_win.h>
#include <falcon/dir_sys_win.h>
#include <falcon/sys.h>
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>
#include <windows.h>
#include <falcon/streambuffer.h>

namespace Falcon 
{

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
      return 0;
   }

   FileStream *fs = new FileStream( new WinFileSysData( handle, 0 ) );
   return new StreamBuffer( fs );
}


Stream *VFSFile::create( const URI& uri, const CParams &p, bool &bSuccess )
{
   DWORD omode = win_paramsToMode( p );
   DWORD oshare = win_paramsToShare( p );
   DWORD ocreate = p.isNoOvr() ? 0 : CREATE_ALWAYS;
   
   // turn the xxx bytes 
   DWORD oattribs = FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_ARCHIVE;
   if ( p.createMode()  )
   {
      // use the owner bits
      int obits = p.createMode() & 0700;

      // set read only if write bit is not set 
      if ( (obits & 0200) == 0 )
      {
         oattribs |= FILE_ATTRIBUTE_READONLY;
      }
      
      // set hidden if read bit is not set
      if ( (obits & 0400) == 0 )
      {
         oattribs |= FILE_ATTRIBUTE_HIDDEN; 
      }
   }

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
      if ( dwError  == ERROR_CALL_NOT_IMPLEMENTED )
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
   }

   if ( handle == 0 || handle == INVALID_HANDLE_VALUE )
   {
      bSuccess = false;
      return 0;
   }

   bSuccess = true;
   // the caller may not really want to open the stream.
   if( p.isNoStream() )
   {
      CloseHandle( handle );
      return 0;
   }
      
   FileStream *fs = new FileStream( new WinFileSysData( handle, 0 ) );
   return new StreamBuffer( fs );
}


DirEntry* VFSFile::openDir( const URI& uri )
{
   int32 error = 0;
   return Sys::fal_openDir( uri.path(), error );
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
   return false;
}


bool VFSFile::chmod( const URI &uri, int mode )
{
   return false;
}

bool VFSFile::link( const URI &uri1, const URI &uri2, bool bSymbolic )
{
   // TODO
   return false;
}


bool VFSFile::unlink( const URI &uri )
{
   int32 err = 0;
   return Sys::fal_unlink( uri.path(), err );
}

bool VFSFile::move( const URI &suri, const URI &duri )
{
   int32 err = 0;
   return Sys::fal_move( suri.path(), duri.path(), err );
}


bool VFSFile::mkdir( const URI &uri, uint32 mode )
{
   int32 err = 0;
   return Sys::fal_mkdir( uri.path(), err );
}


bool VFSFile::rmdir( const URI &uri )
{
   int32 err = 0;
   return Sys::fal_rmdir( uri.path(), err );
}


int64 VFSFile::getLastFsError()
{
   return (int64) GetLastError();
}


Error *VFSFile::getLastError()
{
   DWORD ew = GetLastError();

   if( ew != 0 )
   {
      IoError *e = new IoError( e_io_error );
      e->systemError( ew );
      return e;
   }

   return 0;
}

}

/* end of vsf_file_unix.cpp */
