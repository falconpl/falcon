/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem_private_win.h

   Interprocess shared-memory object private part MS-Windows specific
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_SHMEM_PRIVATE_H
#define FALCON_FEATHERS_SHMEM_PRIVATE_H

#include "sharedmem.h"

#include <falcon/trace.h>
#include <falcon/autowstring.h>
#include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/path.h>
#include <falcon/stderrors.h>

#include <windows.h>

namespace Falcon {


// data in the initial part of the buffer
typedef struct tag_BufferData
{
   int64 size;
   char data[1];
} BufferData;


class SharedMem::Private
{
public:
   HANDLE hFile;
   HANDLE hMemory;
   HANDLE hMtx;

   // Temporary buffer data
   BufferData* bd;
   int64 currentSize;

   String sMemName;

   Private():
	  hFile(INVALID_HANDLE_VALUE),
	  hMemory(INVALID_HANDLE_VALUE),
     hMtx (INVALID_HANDLE_VALUE),
     bd(0),
     currentSize(0)
   {}

   int64 lockAndAlign()
   {
      // ensure we're the only user
      s_lockf();

      // do we really have to update?
      if( currentSize != bd->size )
      {
         currentSize = bd->size;

         if( ! UnmapViewOfFile( bd ) )
         {
            s_unlockf();
            throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                  .extra("UnmapViewOfFile "+ sMemName )
                  .sysError( GetLastError() ) );
         }

         if( ! CloseHandle( hMemory ) )
         {
            s_unlockf();
            throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                              .extra("CreateFileMapping "+ sMemName )
                              .sysError( GetLastError() ) );
         }

          // remap the file.
         AutoWString wMemName( sMemName );
         hMemory = CreateFileMappingW(
            hFile,
            0,
            PAGE_READWRITE,
            0,
            sizeof( BufferData ) +(SIZE_T) currentSize,
            wMemName.w_str() );

         if( hMemory == INVALID_HANDLE_VALUE )
         {
            s_unlockf();
            throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                              .extra("CreateFileMapping "+ sMemName )
                              .sysError( GetLastError() ) );
         }

         bd = (BufferData*) MapViewOfFile(
            hMemory,
            FILE_MAP_WRITE | FILE_MAP_READ,
            0,
            0,
            sizeof( BufferData )+(SIZE_T) currentSize );

         if ( bd == NULL )
         {
            s_unlockf();
            throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                  .extra("MapViewOfFile "+ sMemName )
                  .sysError( GetLastError() ) );
         }
      }

      // DO NOT release the lock
      return currentSize;
   }


   void lockAndResize( int64 newSize )
   {
      // ensure we're the only user
      s_lockf();

      if( newSize == bd->size )
      {
         // nothing needs to be done -- keep the lock
         return;
      }

      enlarge( newSize );

      // DO NOT release the lock
   }

   void enlarge( int64 newSize )
   {
      int64 size = bd->size;

      // fix the expected size now, to avoid problems if we have to drop the process in the middle
      if( newSize < size )
      {
         bd->size = newSize;
      }

      if( ! FlushViewOfFile(bd, (SIZE_T) (size + sizeof(BufferData)) ) )
      {
         s_unlockf();
         throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
               .extra("FlushViewOfFile "+ sMemName )
               .sysError( GetLastError() ) );
      }

      // we'll need to remap with the new size.
      if( ! UnmapViewOfFile( bd ) )
      {
         s_unlockf();
         throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
               .extra("UnmapViewOfFile "+ sMemName )
               .sysError( GetLastError() ) );
      }

      // do we have a background file
      if( hFile != INVALID_HANDLE_VALUE )
      {
         if( ! CloseHandle( hMemory ) )
         {
            s_unlockf();
            throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                  .extra("CloseHandle (mem) "+ sMemName )
                  .sysError( GetLastError() ) );
         }

         // don't leave the pointer dangling.
         hMemory = INVALID_HANDLE_VALUE;

         LONG lHDist = (newSize >> 32) & 0xFFFFFFFF;
         LONG lLDist = newSize & 0xFFFFFFFF;
         // try to resize...
         if( SetFilePointer(hFile, lLDist, &lHDist, FILE_BEGIN ) == INVALID_SET_FILE_POINTER )
         {
            s_unlockf();
            throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                  .extra("SetFilePointer "+ sMemName )
                  .sysError( GetLastError() ) );
         }

         if( ! SetEndOfFile(hFile) )
         {
            // but ingore the error if we're forbidden to shink it.
            DWORD le = GetLastError();
            if( newSize > size || le != ERROR_USER_MAPPED_FILE )
            {
                s_unlockf();
                throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                  .extra("SetEndOfFile "+ sMemName )
                  .sysError( le ) );
            }
            // otherwise, plainly ignore the error
            // -- some process will eventually truncate the file.
         }

         // remap the file.
         AutoWString wMemName( sMemName );
         hMemory = CreateFileMappingW(
            hFile,
            0,
            PAGE_READWRITE,
            0,
            sizeof( BufferData ) + (DWORD) newSize,
            wMemName.w_str() );

         if( hMemory == INVALID_HANDLE_VALUE )
         {
            s_unlockf();
            throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
                              .extra("CreateFileMapping "+ sMemName )
                              .sysError( GetLastError() ) );
         }
      }

      bd = (BufferData*) MapViewOfFile(
         hMemory,
         FILE_MAP_WRITE,
         0,
         0,
         sizeof( BufferData ) + (SIZE_T) newSize );

      if ( bd == NULL )
      {
         s_unlockf();
         throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
               .extra("MapViewOfFile "+ sMemName )
               .sysError( GetLastError() ) );
      }

      // fix the expected size now, to avoid problems if we have to drop the process in the middle
      if( newSize > size )
      {
         bd->size = newSize;
      }

      currentSize = bd->size;
   }


   void s_lockf()
   {
      if( WaitForSingleObject( hMtx, INFINITE ) != WAIT_OBJECT_0 )
      {
         DWORD le = GetLastError();
         TRACE( "SharedMem::Private -- s_lockf ERROR %d", (int)le );
         throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
               .extra("WaitForSingleObject "+ sMemName )
               .sysError( le ) );
      }
   }

   void s_unlockf()
   {
      if( ! ReleaseMutex( hMtx ) )
      {
         DWORD le = GetLastError();
         TRACE( "SharedMem::Private -- s_unlockf ERROR %d", (int)le );
         throw new IOError( ErrorParam( e_io_error, __LINE__, SRC )
               .extra("WaitForSingleObject "+ sMemName )
               .sysError( le ) );
      }

   }

   void close()
   {
      if ( bd != NULL )
      {
         UnmapViewOfFile( bd );
         bd = NULL;
      }

      if( hMemory != INVALID_HANDLE_VALUE )
      {
         CloseHandle( hMemory );
         hMemory = INVALID_HANDLE_VALUE;
      }

      if( hFile != INVALID_HANDLE_VALUE )
      {
         CloseHandle( hFile );
         hFile = INVALID_HANDLE_VALUE;
      }

      if( hMtx != INVALID_HANDLE_VALUE )
      {
         CloseHandle( hMtx );
         hMtx = INVALID_HANDLE_VALUE;
      }
   }
};

}

#endif

/* end of sharedmem_private_win.h */
