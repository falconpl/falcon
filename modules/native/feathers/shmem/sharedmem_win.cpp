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

#include "sharedmem.h"
#include <falcon/autowstring.h>
#include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/path.h>

#include <windows.h>

#define SEM_PREFIX "WOPI_SEM_"
#define APP_PREFIX "WOPI_MEM_"

namespace Falcon {

// data in the initial part of the buffer
typedef struct tag_BufferData
{
   uint32 version;
   int32 size;
} BufferData;

class SharedMemPrivate
{
public:
   SharedMemPrivate():
	  mtx(INVALID_HANDLE_VALUE),
	  hFile(INVALID_HANDLE_VALUE),
	  hMemory(INVALID_HANDLE_VALUE),
     bd(0)
      {}

   HANDLE mtx;
   HANDLE hFile;
   HANDLE hMemory;

   // Temporary buffer data
   BufferData* bd;

   String sMemName;

   void EnterSession()
   {
      if( WaitForSingleObject( this->mtx, INFINITE ) != WAIT_OBJECT_0 )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                            .extra("WaitForSingleObject" )
                                            .sysError( GetLastError() ) );
      }

      // be sure we have the right data in
      this->bd = (BufferData*) MapViewOfFile(
            this->hMemory, FILE_MAP_WRITE, 0, 0, 0 );

      if( this->bd == 0 )
      {
         ReleaseMutex( this->mtx );
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                            .extra("MapViewOfFile" )
                                            .sysError( GetLastError() ) );
      }
   
   }

   void ExitSession()
   {
      UnmapViewOfFile( this->bd );
      this->bd = 0;
      ReleaseMutex( this->mtx );
   }
};



SharedMem::SharedMem( const String &name ):
      d( new SharedMemPrivate ),
      m_version(0)
{
   internal_build( name, "" );
}

SharedMem::SharedMem( const String &name, const String &filename ):
      d( new SharedMemPrivate ),
      m_version(0)
{
   internal_build( name, filename );
}

void SharedMem::internal_build( const String &name, const String &filename )
{
   String sSemName =  SEM_PREFIX + name;
   AutoWString csn( sSemName );

   // Are we the first around?
   bool bFirst = false;
   DWORD dwLastError;

   try
   {
      // try to create the mutex, and take ownership.
      d->mtx = CreateMutexW( 0, TRUE, csn.w_str() );
      if ( d->mtx == INVALID_HANDLE_VALUE )
      {
         
         if ( (dwLastError = GetLastError()) == ERROR_ALREADY_EXISTS )
         {
            // great, the mutex (and the rest) already exists.            
            d->mtx = OpenMutexW( SYNCHRONIZE, FALSE, csn.w_str() );
            if( d->mtx == INVALID_HANDLE_VALUE )
            {
               throw new IoError( ErrorParam( e_io_error, __LINE__ )
                  .extra( "OpenMutex " + sSemName )
                  .sysError( GetLastError() ) );
            }
         }
         else
         {
            throw new IoError( ErrorParam( e_io_error, __LINE__ )
               .extra( "CreateMutex " + sSemName )
               .sysError( dwLastError ) );
         }
      }
      else
      {
         if( GetLastError() != ERROR_ALREADY_EXISTS )
            bFirst = true;
      }

      HANDLE handle;

      // we're the owners of the memory. But, is it new or does it exists?
      if( filename != "" )
      {
         // try to map the file
         Path winName(filename);
         
         AutoWString wfname( winName.getWinFormat() );
         d->hFile = CreateFileW( wfname.w_str(),
             GENERIC_READ | GENERIC_WRITE,
             FILE_SHARE_READ | FILE_SHARE_WRITE,
             0,
             OPEN_ALWAYS,
             FILE_ATTRIBUTE_NORMAL,
             NULL );

         if( d->hFile == INVALID_HANDLE_VALUE )
         {
            throw new IoError( ErrorParam( e_io_error, __LINE__ )
                        .extra("CreateFile "+ filename )
                        .sysError( GetLastError() ) );
         }

         handle = d->hFile;
      }
      else
      {
         handle = INVALID_HANDLE_VALUE;
      }
      

      // create the file mapping.
      d->sMemName = APP_PREFIX + name;
      AutoWString wMemName( d->sMemName );

      d->hMemory = CreateFileMappingW(
            handle,
            0,
            PAGE_READWRITE,
            0,
            sizeof( BufferData ),
            wMemName.w_str() );

      if( d->hMemory == INVALID_HANDLE_VALUE )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
                           .extra("CreateFileMapping "+ d->sMemName )
                           .sysError( GetLastError() ) );
      }

      // ok, let's run -- if we're the first, we should release the mutex
      if( bFirst )
      {
         init();
         ReleaseMutex( d->mtx );
      }
   }
   catch( ... )
   {
      if( bFirst )
      {
         ReleaseMutex( d->mtx );
      }
      close();
      delete d;
      throw;
   }
}


SharedMem::~SharedMem()
{
   close();
   delete d;
}


void SharedMem::init()
{
   // real initialization
   BufferData* bd = (BufferData*) MapViewOfFile(
         d->hMemory,
         FILE_MAP_WRITE,
         0,
         0,
         sizeof( BufferData ) );
      
   if( bd == NULL )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                        .extra("MapViewOfFile" )
                        .sysError( GetLastError() ) );
   }

   if( bd->version == 0 )
   {
      bd->size = 0;
   }
   UnmapViewOfFile( bd );
}


void SharedMem::close()
{
   if ( d->bd != NULL )
   {
      UnmapViewOfFile( d->bd );
      d->bd = NULL;
   }

   if( d->hMemory != INVALID_HANDLE_VALUE )
   {
      CloseHandle( d->hMemory );
      d->hMemory = INVALID_HANDLE_VALUE;
   }

   if( d->hFile != INVALID_HANDLE_VALUE )
   {
      CloseHandle( d->hFile );
      d->hFile = INVALID_HANDLE_VALUE;
   }

   if( d->mtx != INVALID_HANDLE_VALUE )
   {
      CloseHandle( d->mtx );
      d->mtx = INVALID_HANDLE_VALUE;
   }
}


bool SharedMem::read( Stream* target, bool bAlwaysRead )
{
   d->EnterSession();

   // are we aligned?
   if( m_version != d->bd->version && ! bAlwaysRead )
   {
      d->ExitSession();
      return false;
   }

   // align
   try
   {
      internal_read( target );
      d->ExitSession();
   }
   catch( ... )
   {
      d->ExitSession();
      throw;
   }

   return true;
}


bool SharedMem::commit( Stream* source, int32 size, bool bReread  )
{
   // acquire adequate memory mapping.
   d->EnterSession();
      
   if( d->bd == NULL )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                        .extra("MapViewOfFile" )
                        .sysError( GetLastError() ) );
   }

   // are we aligned?
   if( m_version != d->bd->version )
   {
      // ops, we have a problem.
      if( bReread )
      {
         // ok, time to update the data.
         try
         {
            internal_read( source );
         }
         catch( ... )
         {
            d->ExitSession();
            throw;
         }
      }

      d->ExitSession();
      return false;
   }

   if( d->hFile != INVALID_HANDLE_VALUE )
   {
      // try to resize the file
      SetFilePointer( d->hFile, size + sizeof(BufferData), 0, FILE_BEGIN );
      SetEndOfFile( d->hFile );
   }

   // write the new data -- changing the file view
   AutoWString wMemName( d->sMemName );
   HANDLE hView = CreateFileMappingW(
            d->hFile,
            0,
            PAGE_READWRITE,
            0,
            size + sizeof(BufferData),
            wMemName.w_str() );

   if ( hView == INVALID_HANDLE_VALUE )
   {
      d->ExitSession();
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                .extra( String("CreateFileMappingW to ").N(size).A( " bytes" ) )
                                .sysError( GetLastError() ) );
   }


   UnmapViewOfFile( d->bd );
   d->bd = 0;

   void* data = MapViewOfFile( d->hMemory, FILE_MAP_WRITE, 0, 0, 0 );
   if( data == NULL )
   {
      CloseHandle( hView );
      ReleaseMutex( d->mtx );
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                              .extra( String("MapViewOfFile ").N( (int32) size).A(" bytes") )
                              .sysError( GetLastError() ) );
   }

   try
   {
      // ok, read the data from our stream (in the final buffer).
      byte* bdata = ((byte*) data) + sizeof(BufferData);
      int32 written = 0;
      while( written < size )
      {
         int32 rin = source->read( bdata + written, size - written );
         if( rin > 0 )
         {
            written += rin;
         }
         else
         {
            // end of stream?
            if ( rin == 0 )
            {
               size = written;
               break;
            }

            throw new IoError( ErrorParam( e_io_error, __LINE__ )
                           .extra( String("reading from stream") )
                           .sysError( (int32) source->lastError() ) );
         }
      }

      // update the version
      m_version++;
      if( m_version == 0 )
         m_version = 1;
      
      BufferData* bd = (BufferData*) data;
      bd->version = m_version;
      bd->size = size;

      // sync all the buffers, infos and data
      UnmapViewOfFile( data );

      // change the old view with the new one
      CloseHandle( d->hMemory );
      d->hMemory = hView;
      ReleaseMutex( d->mtx );
   }
   catch( ... )
   {
      UnmapViewOfFile( data );
      CloseHandle( hView );
      ReleaseMutex( d->mtx );
      throw;
   }

   return true;
}



void SharedMem::internal_read( Stream* target )
{
   m_version = d->bd->version;
   int32 size = d->bd->size;

   // map the rest of the file
   byte* bdata = (byte*) (d->bd + 1);
   int32 written = 0;
   while( written < size )
   {
      int32 rin = target->write( bdata + written, size - written );
      if( rin > 0 )
      {
         written += rin;
      }
      else
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
                        .extra( String("writing to stream") )
                        .sysError( (int32) target->lastError() ) );
      }
   }
}


uint32 SharedMem::currentVersion() const
{
   d->EnterSession();
   uint32 version = d->bd->version;
   d->ExitSession();

   return version;
}

}

/* end of sharedmem_win.cpp */
