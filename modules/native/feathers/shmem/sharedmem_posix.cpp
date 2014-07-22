/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem_posix.cpp

   Shared memory mapped object.

   Interprocess shared-memory object -- posix part.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/feathers/shmem/sharedmem_posix.cpp"

// To make SUNC happy
#ifdef __SUNPRO_CC
#define _POSIX_C_SOURCE 3
#endif

#include <falcon/autocstring.h>
#include <falcon/stderrors.h>
#include <falcon/stream.h>
#include <falcon/atomic.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>

#include "sharedmem.h"
#include "errors.h"

#define APP_PREFIX "/WPM_"

#include "sharedmem_private_posix.h"

namespace Falcon {

int64 SharedMem::lockAndAlign()
{
   return d->lockAndAlign();
}


void SharedMem::unlock()
{
   d->s_unlockf( FALCON_ERROR_SHMEM_IO_READ );
}


SharedMem::SharedMem():
     d(new Private)
{
}

SharedMem::SharedMem( const String &name, bool bFileBackup ):
      d(new Private)
{
   init( name, true, bFileBackup );
}


void SharedMem::init( const String &name, bool bOpen, bool bFileBackup )
{
   if( ! atomicCAS(d->hasBeenInit, 0, 1) )
   {
      throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_ALREADY_INIT );
   }

   // we're the owners of the memory. But, is it new or does it exists?
   if( bFileBackup )
   {
      // try to map the file
      d->name = name;
      AutoCString cfname( name );
      d->filefd = ::open( cfname.c_str(), O_CREAT | O_RDWR, 0666 );
      if( d->filefd <= 0 )
      {
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT,
                  .extra("open "+ name )
                  .sysError( (int32) errno )
         );
      }
   }
   else
   {
      String sMemName = APP_PREFIX + name;
      d->name = sMemName;
      AutoCString cMemName( sMemName );
      d->shmfd = shm_open( cMemName.c_str(), O_CREAT | O_RDWR, 0666 );
      if( d->shmfd <= 0 )
      {
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT,
                           .extra("shm_open "+ name )
                           .sysError( (int32) errno )
                  );
      }

      d->filefd = d->shmfd;
   }

   int fd = d->filefd;

   off_t pos = 0;
   if( bOpen )
   {
      // ok, we have our stream. See if it needs initialization.
      pos = lseek( fd, 0, SEEK_END );
      if( pos < 0 )
      {
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT,
                                    .extra("lseek" )
                                    .sysError( (int32) errno )
                           );
      }
   }

   // Yes? -- add space.
   if( pos < (int) sizeof(BufferData) )
   {
      if ( ftruncate( fd, sizeof(BufferData) ) != 0 )
      {
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT,
                                          .extra("ftruncate" )
                                          .sysError( (int32) errno )
                                          );
      }
   }

   d->bd = (BufferData*) mmap( 0, sizeof(BufferData), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 );

   if( d->bd == MAP_FAILED )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT,
                                       .extra("mmap" )
                                       .sysError( (int32) errno )
                                       );
   }

   // if the file wasn't empty, we're done
   if ( pos >= (int) sizeof(BufferData) )
   {
      return;
   }

   // real initialization
   d->bd->size = 0;
   d->bd->version = 0;

   if( msync( d->bd, sizeof(BufferData), MS_ASYNC ) != 0 )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT,
                           .extra("msync" )
                           .sysError( (int32) errno )
                           );
   }
}



void SharedMem::close( bool bRemove )
{
   if( d->bd != MAP_FAILED )
   {
      munmap( d->bd, sizeof(BufferData) + d->mapsize );
   }

   int res = 0;
   if( d->shmfd > 0 )
   {
      res = ::close( d->shmfd );
      if( res == 0 && bRemove )
      {
         AutoCString fname(d->name);
         res = ::shm_unlink( fname.c_str() );
      }
   }
   else if( d->filefd > 0 )
   {
      res = ::close( d->filefd );
      if( res == 0 && bRemove )
      {
         AutoCString fname(d->name);
         res = ::unlink(fname.c_str());
      }
   }

   d->shmfd = 0;
   d->filefd = 0;

   if( res != 0)
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                     .extra("close" )
                     .sysError( (int32) errno )
                     );
   }
}


bool SharedMem::internal_write( const void* data, int64 size, int64 offset, bool bSync, bool bTrunc )
{
   // acquire adequate memory mapping. This should work inter-thread as well.
   d->s_lockf( FALCON_ERROR_SHMEM_IO_WRITE );

   // synchronize the version

   // resize the underlying file, if necessary
   // -- if write is larger, or if it's smaller but trunc is required.
   if( (bTrunc && static_cast<int64>(d->bd->size) != static_cast<int64>(offset + size))
            || (static_cast<int64>(offset + size) > static_cast<int64>(d->bd->size)) )
   {
      // actually resize the underlying file
      if ( ftruncate( d->filefd, offset + size + sizeof(BufferData) ) != 0 )
      {
         d->s_unlockf( FALCON_ERROR_SHMEM_IO_WRITE );
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                                   .extra( String("ftruncate to ").N(size).A( " bytes" ) )
                                   .sysError( (int32) errno ) );
      }
   }

   // need to map more space?
   if( d->mapsize < offset + size )
   {
      // unmap, we need to remap the file.
      if( munmap( d->bd, d->mapsize + sizeof(BufferData) ) != 0 )
      {
         d->s_unlockf( FALCON_ERROR_SHMEM_IO_WRITE );
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                                .extra("munmap" )
                                .sysError( (int32) errno )
                                );
      }

      d->bd = (BufferData*) mmap( 0,  offset + size + sizeof(BufferData), PROT_READ | PROT_WRITE, MAP_SHARED, d->filefd, 0 );
      if( d->bd == MAP_FAILED )
      {
         d->s_unlockf( FALCON_ERROR_SHMEM_IO_WRITE );
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                                          .extra("mmap" )
                                          .sysError( (int32) errno )
                                          );
      }


      d->mapsize = d->bd->size = offset + size;
   }


   memcpy( reinterpret_cast<char*>(d->bd)+ offset + sizeof(BufferData), data, size );

   if( msync( d->bd, sizeof(BufferData)+ offset + size, bSync ? MS_SYNC : MS_ASYNC ) != 0 )
   {
      d->s_unlockf( FALCON_ERROR_SHMEM_IO_WRITE );
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                           .extra("msync" )
                           .sysError( (int32) errno )
                           );
   }

   d->s_unlockf( FALCON_ERROR_SHMEM_IO_WRITE );
   return true;
}


int64 SharedMem::localSize() const
{
   return d->mapsize;
}


}

/* end of sharedmem_posix.cpp */
