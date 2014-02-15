/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: fstream_win.cpp

   Windows system specific FILE service support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin:

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/sys.h>
#include <falcon/path.h>
#include <falcon/fstream.h>
#include <falcon/stderrors.h>

#include <windows.h>
#include <wincon.h>

#ifndef INVALID_SET_FILE_POINTER
   #define INVALID_SET_FILE_POINTER ((DWORD)-1)
#endif

namespace Falcon {


FStream::FStream( Sys::FileData* data ):
   m_fsData(data)
{
   m_status = t_open;
   m_bPS = false;
}

FStream::FStream( const FStream &other ):
   Stream( other )
{
   m_fsData = 0;

   Sys::FileData* data = (Sys::FileData*) m_fsData;
   HANDLE hTarget;
   HANDLE hOrig = data->hFile;
   HANDLE curProc = GetCurrentProcess();

   BOOL bRes = ::DuplicateHandle(
                    curProc,
                    hOrig,
                    curProc,
                    &hTarget,
                    0,
                    FALSE,
                    DUPLICATE_SAME_ACCESS);
   if ( ! bRes )
   {
      throw new IOError(
         ErrorParam(e_io_dup, __LINE__, __FILE__ )
         .sysError( ::GetLastError() ) );
   }

   m_fsData = new Sys::FileData( hTarget, data->bIsDiskFile );
}


FStream::~FStream()
{
   close();
   Sys::FileData* data = (Sys::FileData*) m_fsData;
   delete data;
}


bool FStream::close()
{
   Sys::FileData* data = (Sys::FileData*) m_fsData;
   HANDLE hFile = data->hFile;

   if ( m_status & Stream::t_open )
   {
      if( ! ::CloseHandle( hFile ) )
      {
         m_lastError = (size_t) GetLastError();
         m_status = m_status | t_error;

         if( m_bShouldThrow )
         {
            throw new IOError( ErrorParam( e_io_close, __LINE__, __FILE__ ).sysError( GetLastError() ) );
         }

         return false;
      }
   }

   m_lastError = 0;
   m_status = m_status & (~ Stream::t_open);
   return true;
}


size_t FStream::read( void *buffer, size_t size )
{
   Sys::FileData* data = (Sys::FileData*) m_fsData;
   HANDLE hFile = data->hFile;

   DWORD result;
   if ( ! ::ReadFile( hFile, buffer, size, &result, NULL ) )
   {
      m_lastError = (size_t) ::GetLastError();
      m_status = m_status | t_error;

      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_read, __LINE__, __FILE__ ).sysError( GetLastError() ) );
      }

      return -1;
   }

   if ( result == 0 ) {
      m_status = m_status | Stream::t_eof;
   }

   m_lastError = 0;
   return (size_t) result;
}

size_t FStream::write( const void *buffer, size_t size )
{
   Sys::FileData* data = (Sys::FileData*) m_fsData;
   HANDLE hFile = data->hFile;

   DWORD result;
   if ( ! ::WriteFile( hFile, buffer, size, &result, NULL ) )
   {
      m_lastError = (size_t) ::GetLastError();
      m_status = m_status | t_error;

      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_write, __LINE__, __FILE__ )
                    .sysError( ::GetLastError() ) );
      }

      return false;
   }

   m_lastError = 0;
   return (size_t) result;
}


off_t FStream::seek( off_t pos, e_whence whence )
{
   Sys::FileData* data = (Sys::FileData*) m_fsData;
   HANDLE hFile = data->hFile;

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

   DWORD npos = (int32) SetFilePointer( hFile, posLow, &posHI, from );
   if( npos == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR ) {
      m_lastError = ::GetLastError();
      m_status = m_status | Stream::t_error;
      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_write, __LINE__, __FILE__ )
                    .sysError( ::GetLastError() ) );
      }

      return -1;
   }
   else {
#ifdef _MSC_VER
      m_status = (Stream::t_status)(((uint32)m_status) & ~((uint32)Stream::t_eof));
#else
      m_status = m_status & ~Stream::t_eof;
#endif
   }

   m_lastError = 0;
   return npos | ((int64)posHI) << 32;
}


off_t FStream::tell()
{
   Sys::FileData* data = (Sys::FileData*) m_fsData;
   HANDLE hFile = data->hFile;

   LONG posHI = 0;

   DWORD npos = (int32) SetFilePointer( hFile, 0, &posHI, FILE_CURRENT );
   if( npos == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR ) {
      m_lastError = ::GetLastError();
      m_status = m_status | Stream::t_error;
      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_write, __LINE__, __FILE__ )
                    .sysError( ::GetLastError() ) );
      }

      return -1;
   }

   m_lastError = 0;
   return npos | ((int64)posHI) << 32;
}


bool FStream::truncate( off_t pos )
{
   Sys::FileData* data = (Sys::FileData*) m_fsData;
   HANDLE hFile = data->hFile;
   LONG savePosHI = 0;
   LONG savePosLow = 0;
   off_t oldPos = 0;

   if ( pos >= 0 )
   {

      LONG posHI = (LONG) (pos >> 32);
      LONG posLow = (LONG)(pos & 0xFFFFFFFF);

      // Get current position.
      savePosLow = SetFilePointer( hFile, 0, &savePosHI, FILE_CURRENT );
      if( ::GetLastError() != NO_ERROR )
      {
         goto on_error;
      }
      oldPos = posHI;
      oldPos <<= 32;
      oldPos |= posLow;

      SetFilePointer( hFile, posLow, &posHI, FILE_BEGIN );
      if( ::GetLastError() != NO_ERROR )
      {
         goto on_error;
      }
   }

   if( ! SetEndOfFile( hFile ) )
   {
      goto on_error;
   }

   // Need to move the file pointer back?
   if( pos > oldPos )
   {
      SetFilePointer( hFile, savePosLow, &savePosHI, FILE_BEGIN );
      if( ::GetLastError() != NO_ERROR )
      {
         goto on_error;
      }
   }

   m_lastError = 0;
   return true;

on_error:
   m_lastError = ::GetLastError();
   m_status = m_status | Stream::t_error;

   if( m_bShouldThrow )
   {
      throw new IOError( ErrorParam( e_io_write, __LINE__, __FILE__ )
                 .sysError( ::GetLastError() ) );
   }
   return false;
}

}

/* end of fstream_win.cpp */
