/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: fstream_sys_win.cpp

   Unix system specific FILE service support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Widnows system specific FILE service support.
*/

#include <falcon/fstream_sys_win.h>
#include <falcon/memory.h>
#include <falcon/sys.h>
#include <falcon/vm_sys_win.h>
#include <falcon/path.h>

#include <wincon.h>

#ifndef INVALID_SET_FILE_POINTER
   #define INVALID_SET_FILE_POINTER ((DWORD)-1)
#endif

namespace Falcon {

FileSysData *WinFileSysData::dup()
{
   HANDLE dupped;
   HANDLE curProc = GetCurrentProcess();

   if (! DuplicateHandle(
        curProc,  // handle to the source process
        m_handle,         // handle to duplicate
        curProc,  // handle to process to duplicate to
        &dupped,  // pointer to duplicate handle
        0,    // access for duplicate handle
        TRUE,      // handle inheritance flag
        DUPLICATE_SAME_ACCESS           // optional actions
         )
         )
   {
      return 0;
   }

   return new WinFileSysData( dupped, m_lastError, m_isConsole, m_direction, m_isPipe );
}

BaseFileStream::BaseFileStream( const BaseFileStream &gs ):
   Stream( gs )
{
   m_fsData = gs.m_fsData->dup();
}

BaseFileStream::~BaseFileStream()
{
   close();
   delete m_fsData;
}

bool BaseFileStream::close()
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   if ( open() ) {
      if( ! CloseHandle( data->m_handle ) ) {
         data->m_lastError = GetLastError();
         m_status = Stream::t_error;
         return false;
      }
   }

   data->m_lastError = 0;
   m_status = m_status & static_cast<Stream::t_status>(~
                  static_cast<unsigned int>(Stream::t_open));
   return true;
}

int32 BaseFileStream::read( void *buffer, int32 size )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   DWORD result;
   if ( ! ReadFile( data->m_handle, buffer, size, &result, NULL ) ) {
      data->m_lastError = GetLastError();
      if( data->m_lastError == ERROR_NOACCESS || 
         data->m_lastError == ERROR_HANDLE_EOF ||
         data->m_lastError == ERROR_BROKEN_PIPE )
      {
         // ReadFile returns ERROR_NOACCESS at EOF
         data->m_lastError = 0;
         m_status = m_status | Stream::t_eof;
         return 0;
      }

      m_status = m_status | Stream::t_error;
      return -1;
   }

   if ( result == 0 ) 
   {
      m_status = m_status | Stream::t_eof;
   }

   data->m_lastError = 0;
   m_lastMoved = result;
   return result;
}

int32 BaseFileStream::write( const void *buffer, int32 size )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   DWORD result;
   if ( ! WriteFile( data->m_handle, buffer, size, &result, NULL ) ) {
      data->m_lastError = GetLastError();
      m_status = m_status | Stream::t_error;
      return -1;
   }

   data->m_lastError = 0;
   m_lastMoved = result;
   return result;
}

bool BaseFileStream::put( uint32 chr )
{
   byte b = (byte) chr;
   return write( &b, 1 ) == 1;
}

bool BaseFileStream::get( uint32 &chr )
{
   if( popBuffer( chr ) )
      return true;

   byte b;
   if ( read( &b, 1 ) == 1 )
   {
      chr = (uint32) b;
      return true;
   }
   return false;
}


int64 BaseFileStream::seek( int64 pos, e_whence whence )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   DWORD from;
   switch(whence) {
      case 0: from = FILE_BEGIN; break;
      case 1: from = FILE_CURRENT; break;
      case 2: from = FILE_END; break;
      default:
         from = FILE_BEGIN;
   }

   LONG posLow = (LONG)(pos & 0xFFFFFFFF);
   LONG posHI = (LONG) (pos >> 32);

   DWORD npos = (int32) SetFilePointer( data->m_handle, posLow, &posHI, from );
   if( npos == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR ) {
      data->m_lastError = GetLastError();
      m_status = m_status | Stream::t_error;
      return -1;
   }
   else {
#ifdef _MSC_VER
      m_status = (Stream::t_status)(((uint32)m_status) & ~((uint32)Stream::t_eof));
#else
      m_status = m_status & ~Stream::t_eof;
#endif
   }

   data->m_lastError = 0;
   return npos | ((int64)posHI) << 32;
}


int64 BaseFileStream::tell()
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   LONG posHI = 0;

   DWORD npos = (int32) SetFilePointer( data->m_handle, 0, &posHI, FILE_CURRENT );
   if( npos == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR ) {
      data->m_lastError = GetLastError();
      m_status = m_status | Stream::t_error;
      return -1;
   }

   data->m_lastError = 0;
   return npos | ((int64)posHI) << 32;
}


bool BaseFileStream::truncate( int64 pos )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   if ( pos > 0 )
   {
      LONG posLow = (LONG)(pos & 0xFFFFFFFF);
      LONG posHI = (LONG) (pos >> 32);

      SetFilePointer( data->m_handle, posLow, &posHI, FILE_BEGIN );
      if( GetLastError() != NO_ERROR ) {
         data->m_lastError = GetLastError();
         m_status = m_status | Stream::t_error;
         return false;
      }
   }

   SetEndOfFile( data->m_handle );
   if( GetLastError() != NO_ERROR ) {
      data->m_lastError = GetLastError();
      m_status = m_status | Stream::t_error;
      return false;
   }

   SetFilePointer( data->m_handle, 0, 0, FILE_END );

   data->m_lastError = 0;
   return true;
}

bool BaseFileStream::errorDescription( ::Falcon::String &description ) const
{
	if ( Stream::errorDescription( description ) )
      return true;

	WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );
   if ( data->m_lastError == 0 )
      return false;

	LPVOID lpMsgBuf;

   DWORD res = FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM,
      0,
      data->m_lastError,
      LANG_USER_DEFAULT,
      (LPTSTR) &lpMsgBuf,
      0,
      NULL
    );

   if ( res == 0 ) {
      description = "Impossible to retreive error description";
		return false;
   }
   else
   {
      description = (char *) lpMsgBuf;
		description.bufferize();
		LocalFree(lpMsgBuf);
		return true;
   }

}

int64 BaseFileStream::lastError() const
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );
   return (int64) data->m_lastError;
}


int BaseFileStream::readAvailable( int32 msec, const Sys::SystemData *sysData )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   if ( data->m_isConsole || data->m_isPipe ) {

      if ( data->m_direction == WinFileSysData::e_dirOut )
      {
         if (msec > 0 )
            ::Sleep( msec );
         return false;
      }

      DWORD waitTime;
      DWORD size = 0;

      if ( data->m_isPipe )
      {
         if ( msec < 0 ) {
            while ( size == 0 ) {
               Sleep(10);
               if ( ! PeekNamedPipe( data->m_handle, NULL, 0,  NULL, &size, NULL ) )
                  return -1;
            }
            return 1;
         }
         else {
            waitTime = (DWORD) msec;

            DWORD result = PeekNamedPipe( data->m_handle, NULL, 0,  NULL, &size, NULL );
            while ( result && size == 0 && waitTime > 0 )
            {
               // Interrupted?
               if( WaitForSingleObject( sysData->m_sysData->evtInterrupt, 1 ) == WAIT_OBJECT_0 )
                  return -2;

               waitTime -= 1;
               result = PeekNamedPipe( data->m_handle, NULL, 0,  NULL, &size, NULL );
            }
            if ( ! result )
               return -1;
            return (size > 0) ? 1 : 0;
         }
      }
      else
      {
         waitTime = msec < 0 ? INFINITE : msec;
         HANDLE waiting[2];
         waiting[0] = data->m_handle;
         waiting[1] = sysData->m_sysData->evtInterrupt;
         DWORD res = WaitForMultipleObjects( 2, waiting, FALSE,waitTime );

         if ( res == WAIT_OBJECT_0 )
            return 1;

         // Interrupted?
         if ( res == WAIT_OBJECT_0 + 1 )
            return -2;
      }

      return 0;
   }

   // on windows, disk files are always available.
   return 1;
}

int32 BaseFileStream::writeAvailable( int32 msec, const Sys::SystemData *sysData )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   if ( data->m_isConsole || data->m_isPipe ) {

      if ( data->m_direction == WinFileSysData::e_dirIn )
      {
         if (msec > 0 )
         {
            if ( sysData->sleep( msec ) )
               return 0;
            return -2; // interrupted
         }
      }

      //DWORD waitTime;
      DWORD size = 0;

      if ( data->m_isPipe )
      {
         // no way to know, by now
      }
      else {
         /* Always ready on windows.
         waitTime = msec < 0 ? INFINITE : msec;
         if ( WaitForSingleObject( data->m_handle, waitTime ) == WAIT_OBJECT_0 ) {
            return 1;
         }*/
      }

      if ( sysData->interrupted() )
         return -2;

      return 1;
   }

   // on windows, disk files are always available.
   return 1;
}

bool BaseFileStream::writeString( const String &content, uint32 begin, uint32 end )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );
   uint32 done = begin;
   uint32 stop = content.size();
   uint32 charSize = content.manipulator()->charSize();
   if ( end < stop / charSize )
      stop = end * charSize;

   while ( done < stop )
   {
      DWORD result;
      if (! WriteFile( data->m_handle, content.getRawStorage() + done, content.size() - done, &result, NULL ) )
      {
         setError( GetLastError() );
         m_lastMoved = done;
         return false;
      }
      done += result;
   }

   setError( 0 );
   m_lastMoved = done - begin;
   return true;
}


bool BaseFileStream::readString( String &content, uint32 size )
{
   // TODO OPTIMIZE
   uint32 chr;
   content.size( 0 );
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );

   while( size > 0 && get( chr ) )
   {
      size--;
      content.append( chr );
   }

   if ( size == 0 || eof() )
   {
      setError( 0 );
      return true;
   }

   setError( errno );
   return false;
}

void BaseFileStream::setError( int64 errorCode )
{
   WinFileSysData *data = static_cast< WinFileSysData *>( m_fsData );
   data->m_lastError = (uint32) errorCode;
   if ( errorCode != 0 )
      status( status() | t_error );
   else
      status( (t_status) (((int)status()) & ~(int)Stream::t_error ));
}

BaseFileStream *BaseFileStream::clone() const
{
   BaseFileStream *gs = new BaseFileStream( *this );
   if ( gs->m_fsData == 0 )
   {
      delete gs;
      return 0;
   }

   return gs;
}

//========================================================
// File stream
//

FileStream::FileStream():
   BaseFileStream( t_file, new WinFileSysData( INVALID_HANDLE_VALUE, 0 ) )
{
   status( t_none );
}

bool FileStream::open( const String &filename_flc, t_openMode mode, t_shareMode share )
{
   WinFileSysData *data = static_cast< WinFileSysData * >( m_fsData );
   DWORD omode = 0;
	DWORD shmode = 0;

   /*if ( data->m_handle != 0 )
      CloseHandle( data->m_handle );*/

   if ( mode == e_omReadWrite )
      omode = GENERIC_READ | GENERIC_WRITE;
   else if ( mode == e_omReadOnly )
      omode = GENERIC_READ;
   else
      omode = GENERIC_WRITE;

	if( share == e_smShareRead )
		shmode = FILE_SHARE_READ;
	else if ( share == e_smShareFull )
		shmode = FILE_SHARE_READ | FILE_SHARE_WRITE;

   // todo: do something about share mode
	String filename = filename_flc;
   Path::uriToWin( filename );

	uint32 bufsize = filename.length() * sizeof( wchar_t ) + sizeof( wchar_t );

   wchar_t *buffer = ( wchar_t *) memAlloc( bufsize );
   if (buffer == 0) {
      setError( -2 );
      return false;
   }

   filename.toWideString( buffer, bufsize );
	HANDLE handle = CreateFileW( buffer,
      omode,
      shmode,
      NULL,
      OPEN_EXISTING,
      0,
      NULL );

	DWORD dwError = GetLastError();
   if ( handle == 0 || handle == INVALID_HANDLE_VALUE )
   {
      if ( dwError  == ERROR_CALL_NOT_IMPLEMENTED )
      {
         char *charbuf = (char *) buffer;
         if( filename.toCString( charbuf, bufsize ) > 0 )
         {
            handle = CreateFile( charbuf,
               omode,
               shmode,
               NULL,
               OPEN_EXISTING,
               0,
               NULL );
         }
      }
   }

	memFree( buffer );

   data->m_handle = handle;
   if ( handle == 0 || handle == INVALID_HANDLE_VALUE)
   {
      setError( GetLastError() );
      status( t_error );
      return false;
   }

   status( t_open );
   data->m_lastError = 0;
   return true;
}

bool FileStream::create( const String &filename_flc, t_attributes mode, t_shareMode share )
{
   WinFileSysData *data = static_cast< WinFileSysData * >( m_fsData );
   DWORD shmode = 0;

   if ( data->m_handle > 0 )
      CloseHandle( data->m_handle );

	// for now ignore attributes

	if( share == e_smShareRead )
		shmode = FILE_SHARE_READ;
	else if ( share == e_smShareFull )
		shmode = FILE_SHARE_READ | FILE_SHARE_WRITE;

   // todo: do something about share mode
	String filename = filename_flc;
   Path::uriToWin( filename );

	uint32 bufsize = filename.length() * sizeof( wchar_t ) + sizeof( wchar_t );
   wchar_t *buffer = ( wchar_t *) memAlloc( bufsize );
   if (buffer == 0) {
      setError( -2 );
      return false;
   }

   filename.toWideString( buffer, bufsize );

	HANDLE handle = CreateFileW( buffer,
      GENERIC_READ | GENERIC_WRITE,
      shmode,
      NULL,
      CREATE_ALWAYS,
      FILE_ATTRIBUTE_ARCHIVE,
      NULL );

	if ( handle == 0 || handle == INVALID_HANDLE_VALUE )
   {
      if ( GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
      {
         char *charbuf = (char *) buffer;
         if( filename.toCString( charbuf, bufsize ) > 0 )
         {
            handle = CreateFile( charbuf,
               GENERIC_READ | GENERIC_WRITE,
               shmode,
               NULL,
               CREATE_ALWAYS,
               FILE_ATTRIBUTE_ARCHIVE,
               NULL );
         }
      }
   }
	memFree( buffer );

   data->m_handle = handle;
   if ( handle == 0 || handle == INVALID_HANDLE_VALUE ) {
      setError( GetLastError() );
      status( t_error );
      return false;
   }

   status( t_open );
   data->m_lastError = 0;
   return true;
}

void FileStream::setSystemData( const FileSysData &fsData )
{
   const WinFileSysData *data = static_cast< const WinFileSysData *>( &fsData );
   WinFileSysData *myData = static_cast< WinFileSysData *>( m_fsData );
   myData->m_handle = data->m_handle;
   myData->m_lastError = data->m_lastError;
   myData->m_direction = data->m_direction;
   myData->m_isConsole = data->m_isConsole;
   myData->m_isPipe = data->m_isPipe;
}


StdInStream::StdInStream():
   InputStream( 0 )
{
   HANDLE dupped;
   HANDLE curProc = GetCurrentProcess();

   if (! DuplicateHandle(
        curProc,  // handle to the source process
        GetStdHandle( STD_INPUT_HANDLE ),         // handle to duplicate
        curProc,  // handle to process to duplicate to
        &dupped,  // pointer to duplicate handle
        GENERIC_READ,    // access for duplicate handle
        FALSE,      // handle inheritance flag
        0           // optional actions
         )
      )
   {
      dupped = GetStdHandle( STD_INPUT_HANDLE );
      if( dupped != INVALID_HANDLE_VALUE )
      {
         SetConsoleMode( dupped, 0 );

         /*
         SetConsoleMode( dupped,
			   ENABLE_ECHO_INPUT |
			   ENABLE_PROCESSED_INPUT );
            */
      }
   }
   else {
      SetConsoleMode( dupped, 0 );
	   // remove the readline mode, as it breaks the WaitForXX functions.
	   /*SetConsoleMode( dupped,
			ENABLE_ECHO_INPUT |
			ENABLE_PROCESSED_INPUT );*/
   }

   m_fsData = new WinFileSysData( dupped, 0, true, WinFileSysData::e_dirIn );
}

StdOutStream::StdOutStream():
   OutputStream( 0 )
{
   HANDLE dupped;
   HANDLE curProc = GetCurrentProcess();

   if (! DuplicateHandle(
        curProc,  // handle to the source process
        GetStdHandle( STD_OUTPUT_HANDLE ),         // handle to duplicate
        curProc,  // handle to process to duplicate to
        &dupped,  // pointer to duplicate handle
        GENERIC_WRITE,    // access for duplicate handle
        FALSE,      // handle inheritance flag
        0           // optional actions
         )
      )
   {
      dupped = GetStdHandle( STD_OUTPUT_HANDLE );
   }

   m_fsData = new WinFileSysData( dupped, 0, true, WinFileSysData::e_dirOut );
}

StdErrStream::StdErrStream():
   OutputStream( 0 )
{
   HANDLE dupped;
   HANDLE curProc = GetCurrentProcess();

   if (! DuplicateHandle(
        curProc,  // handle to the source process
        GetStdHandle( STD_ERROR_HANDLE ),         // handle to duplicate
        curProc,  // handle to process to duplicate to
        &dupped,  // pointer to duplicate handle
        GENERIC_WRITE,    // access for duplicate handle
        FALSE,      // handle inheritance flag
        0           // optional actions
         )
      )
   {
      dupped = GetStdHandle( STD_ERROR_HANDLE );
   }

   m_fsData = new WinFileSysData( dupped, 0, true, WinFileSysData::e_dirOut );
}


RawStdInStream::RawStdInStream():
InputStream( new WinFileSysData( GetStdHandle( STD_INPUT_HANDLE ), 0, true, WinFileSysData::e_dirIn ) )
{
}

RawStdOutStream::RawStdOutStream():
   OutputStream( new WinFileSysData( GetStdHandle( STD_OUTPUT_HANDLE ), 0, true, WinFileSysData::e_dirOut ) )
{
}

RawStdErrStream::RawStdErrStream():
   OutputStream( new WinFileSysData( GetStdHandle( STD_ERROR_HANDLE ), 0, true, WinFileSysData::e_dirOut ) )
{
}


}


/* end of file_srv_unix.cpp */
