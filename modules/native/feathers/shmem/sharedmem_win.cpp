/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem_win.cpp

   Interprocess shared-memory object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/feathers/shmem/sharedmem_win.cpp"

#include "sharedmem.h"
#include "errors.h"
#include <falcon/trace.h>
#include <falcon/autowstring.h>
#include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/path.h>
#include <falcon/stderrors.h>

#include <windows.h>

#include "sharedmem_private_win.h"

namespace Falcon {

SharedMem::SharedMem():
      d( new Private )
{
}


SharedMem::SharedMem( const String &name, bool bFileBackup ):
      d(new Private)
{
   init( name, true, bFileBackup );
}


void SharedMem::init( const String &name, bool bOpen, bool bFileBackup )
{
   if( bFileBackup )
   {
      // try to map the file
      Path winName(name);
      AutoWString wfname( winName.getWinFormat() );
      d->hFile = CreateFileW( wfname.w_str(),
          GENERIC_READ | GENERIC_WRITE,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          0,
          (bOpen ? OPEN_ALWAYS : CREATE_ALWAYS),
          FILE_ATTRIBUTE_NORMAL,
          NULL );

      if( d->hFile == INVALID_HANDLE_VALUE )
      {
         throw new IOError( ErrorParam( e_io_error, __LINE__ )
                     .extra("CreateFile "+ name )
                     .sysError( GetLastError() ) );
      }

      // if we have open semantic, ensure there is enough space.
      LONG result = 0;
      LONG pl = 0;
      if( bOpen )
      {
         result = SetFilePointer( d->hFile, 0, &pl, FILE_END );
      }

      // smaller than needed? -- or just created?
      if( result < (LONG) sizeof(BufferData) && pl == 0 )
      {
         // reset its size.
         SetFilePointer( d->hFile, 0, &pl, FILE_BEGIN );
         BufferData bd;
         ZeroMemory(&bd, sizeof(bd));
         DWORD dwWriteCount = 0;
         WriteFile( d->hFile, &bd, sizeof(bd), &dwWriteCount, NULL );
         // Ignore errors: they might be caused by concurrent processes doing the same thing,
         // or by I/O errors that we'll see later on.
      }
   }
   else
   {
      d->hFile = INVALID_HANDLE_VALUE;
   }

   // create the file mapping.
   d->sMemName = name;
   AutoWString wMemName( d->sMemName );
   AutoWString wMtxMemName( "MTX_" + d->sMemName );
   d->hMtx = CreateMutexW( NULL, FALSE, wMtxMemName.w_str() );

   if( d->hMtx == NULL || d->hMtx == INVALID_HANDLE_VALUE )
   {
      DWORD le = GetLastError();
      TRACE("SharedMem::init -- CreateMutexW failed because %d", (int)le );
      if( le == ERROR_ALREADY_EXISTS )
      {
         TRACE("SharedMem::init -- Trying to open %d", (int)le );
         d->hMtx = OpenMutexW( SYNCHRONIZE|MUTEX_MODIFY_STATE, FALSE, wMtxMemName.w_str() );
      }

      if( d->hMtx == NULL || d->hMtx == INVALID_HANDLE_VALUE )
      {
         TRACE("SharedMem::init -- definitive failure %d", (int)le );
         throw new IOError( ErrorParam( e_io_error, __LINE__ )
                        .extra("CreateFileMapping "+ d->sMemName )
                        .sysError( le ) );
      }
   }

   // try to open the mapping
   // with create semantic, we must rewrite the header, if present.
   d->hMemory = CreateFileMappingW(
         d->hFile,
         0,
         PAGE_READWRITE,
         0,
         sizeof( BufferData ),
         wMemName.w_str() );

   if( d->hMemory == NULL || d->hMemory == INVALID_HANDLE_VALUE )
   {
      DWORD le = GetLastError();
      TRACE("SharedMem::init -- CreateFileMappingW failed because %d", (int)le );
      throw new IOError( ErrorParam( e_io_error, __LINE__ )
                        .extra("CreateFileMapping "+ d->sMemName )
                        .sysError( le ) );
   }

   //DWORD alreadyCreated = GetLastError();

   // correctly opened -- or created. Let's map it.
   d->bd = (BufferData*) MapViewOfFile(
      d->hMemory,
      FILE_MAP_WRITE | FILE_MAP_READ,
      0,
      0,
      sizeof( BufferData ) );

   if( d->bd == NULL )
   {
      DWORD le = GetLastError();
      TRACE("SharedMem::init -- MapViewOfFile failed because %d", (int)le );
      throw new IOError( ErrorParam( e_io_error, __LINE__ )
                           .extra("MapViewOfFile "+ d->sMemName )
                           .sysError( le ) );
   }

   // We don't need to update the size; the first I/O will do.
}


void SharedMem::close( bool bRemove )
{
   //bool rem = bRemove && d->hFile != INVALID_HANDLE_VALUE;

   d->close();
   if( bRemove )
   {
      // ignore result
      AutoWString wname(d->sMemName);
      DeleteFileW( wname.w_str() );
   }
}


int64 SharedMem::localSize() const
{
   return d->currentSize;
}


int64 SharedMem::lockAndAlign()
{
   return d->lockAndAlign();
}


void SharedMem::unlock()
{
   d->s_unlockf();
}

bool SharedMem::internal_write( const void* data, int64 size, int64 offset, bool bSync, bool bTrunc )
{
   if( size == 0 )
   {
      return true;
   }

   if( bTrunc )
   {
      d->lockAndResize( offset+size );
   }
   else {
      int64 curSize = d->lockAndAlign();

      if( offset + size > curSize )
      {
         d->enlarge( offset + size );
      }
   }

   memcpy( static_cast<char*>(d->bd->data) + offset, data, (size_t) size );
   if( bSync && d->hFile != INVALID_HANDLE_VALUE )
   {
      FlushViewOfFile( d->bd, (SIZE_T) (offset + size +sizeof(d->bd)) );
   }

   d->s_unlockf();

   return true;
}

}

/* end of sharedmem_win.cpp */
