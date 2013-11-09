/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem_posix.cpp

   Shared memory mapped object.

   TODO: Move this file in the main engine in the next version.

   Interprocess shared-memory object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/feathers/shmem/sharedmem_posix.cpp"

// To make SUNC happy
#define _POSIX_C_SOURCE 3

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

namespace Falcon {

// data in the initial part of the buffer
typedef struct tag_BufferData
{
   uint32 version;
   size_t size;
} BufferData;

class SharedMem::Private
{
public:
   Private():
      hasBeenInit(0),
      shmfd(0),
      filefd(0),
      version(0),
      mapsize(0),
      cache(MAP_FAILED),
      bd(static_cast<BufferData*>(MAP_FAILED))
      {}

   atomic_int hasBeenInit;
   int shmfd;
   int filefd;
   uint32 version;
   int64 mapsize;
   void* cache;

   // Memory mapped data
   BufferData* bd;
};


SharedMem::SharedMem():
     d(new Private)
{
}

SharedMem::SharedMem( const String &name, bool bFileBackup ):
      d(new Private)
{
   init( name, bFileBackup );
}


void SharedMem::init( const String &name, bool bFileBackup )
{
   if( ! atomicCAS(d->hasBeenInit, 0, 1) )
   {
      throw FALCON_SIGN_ERROR(ShmemError, FALCON_ERROR_SHMEM_ALREADY_INIT );
   }

   // we're the owners of the memory. But, is it new or does it exists?
   if( bFileBackup )
   {
      // try to map the file
      AutoCString cfname( name );
      d->filefd = open( cfname.c_str(), O_CREAT | O_RDWR, 0666 );
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

   // ok, we have our stream. See if it needs initialization.
   off_t pos = lseek( fd, 0, SEEK_END );
   if( pos < 0 )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_INIT,
                                 .extra("lseek" )
                                 .sysError( (int32) errno )
                        );
   }

   // Yes? -- add space.
   if( pos == 0 )
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
   if ( pos != 0 )
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


SharedMem::~SharedMem()
{
   try {
      this->close();
   }
   catch(Error* err )
   {
      err->decref();
   }
   catch( ... )
   {}

   delete d;
}


void SharedMem::close()
{
   if( d->bd != MAP_FAILED )
   {
      munmap( d->bd, sizeof(BufferData) );
   }

   if( d->cache != MAP_FAILED )
   {
      munmap( d->cache, d->mapsize );
   }

   int res;
   if( d->shmfd > 0 )
   {
      res = ::close( d->shmfd );
   }
   else if( d->filefd > 0 )
   {
      res = ::close( d->filefd );
   }

   d->shmfd = 0;
   d->filefd = 0;
   d->cache = MAP_FAILED;

   if( res != 0)
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                     .extra("close" )
                     .sysError( (int32) errno )
                     );
   }
}


bool SharedMem::read( void* data, int64& size, int64 offset )
{
   // acquire adequate memory mapping. This should work inter-thread as well.
   if( lockf(d->filefd, F_LOCK, 0 ) != 0 )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                  .extra("lockf" )
                  .sysError( (int32) errno )
                  );
   }

   // d->bd is mem-mapped; any change is shared.
   int64 rdSize = d->bd->size;
   int fd = d->filefd;

   // a valid request?
   bool retval;
   if( (size+offset) >= rdSize )
   {
      if ( offset >= rdSize )
      {
         size = 0;
      }
      else
      {
         // are we aligned?
         if( d->cache != MAP_FAILED && d->mapsize != d->bd->size  )
         {
            munmap( d->cache, d->mapsize );
            d->cache = MAP_FAILED;
         }

         // msync if necessary
         if( d->cache == MAP_FAILED )
         {
            // map the rest of the file -- possibly change the cache.
            d->cache = mmap( d->cache, rdSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, sizeof(BufferData) );

            if( d->cache == MAP_FAILED )
            {
               lockf(d->filefd, F_UNLCK, 0 );
               throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                              .extra("mmap" )
                              .sysError( (int32) errno )
                              );
            }

            d->mapsize = rdSize;
            d->version = d->bd->version;
         }

         // according to POSIX, d->cache is directly visible in all the processes.
         size = rdSize - offset;
         memcpy( data, static_cast<char*>(d->cache) + offset, size );
      }

      retval = true;
   }
   else
   {
      retval = false;
      // update with the real size.
      size = rdSize;
   }


   // release the file lock
   if( lockf(d->filefd, F_UNLCK, 0 ) != 0 )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                  .extra("lockf - unlock" )
                  .sysError( (int32) errno )
                  );
   }

   return retval;
}



void* SharedMem::grab( void* data, int64& size, int64 offset )
{
   // acquire adequate memory mapping. This should work inter-thread as well.
   if( lockf(d->filefd, F_LOCK, 0 ) != 0 )
   {
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                  .extra("lockf" )
                  .sysError( (int32) errno )
                  );
   }

   // d->bd is mem-mapped; any change is shared.
   int64 rdSize = d->bd->size;
   int fd = d->filefd;

   // a valid request?
   if ( offset >= rdSize )
   {
      size = 0;
      return data;
   }

   // are we aligned?
   if( d->cache != MAP_FAILED && d->mapsize != d->bd->size  )
   {
      munmap( d->cache, d->mapsize );
      d->cache = MAP_FAILED;
   }

   // msync if necessary
   if( d->cache == 0 )
   {
      // map the rest of the file -- possibly change the cache.
      d->cache = mmap( d->cache, rdSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, sizeof(BufferData) );

      if( d->cache == MAP_FAILED )
      {
         lockf(d->filefd, F_UNLCK, 0 );
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                        .extra("mmap" )
                        .sysError( (int32) errno )
                        );
      }
      d->mapsize = rdSize;
      d->version = d->bd->version;
   }

   // according to POSIX, d->cache is directly visible in all the processes.

   // resize the data if needed.
   bool bNewData = false;
   if ( size < rdSize )
   {
      bNewData = true;
      data = malloc(rdSize);
   }

   size = rdSize - offset;
   memcpy( data, static_cast<char*>(d->cache) + offset, size );

   // release the file lock
   if( lockf(d->filefd, F_UNLCK, 0 ) != 0 )
   {
      if( bNewData )
      {
         free(data);
      }

      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                  .extra("lockf - unlock" )
                  .sysError( (int32) errno )
                  );
   }

   return data;
}


bool SharedMem::internal_write( const void* data, int64 size, int64 offset, bool bSync, bool bTry, bool bTrunc )
{
   // acquire adequate memory mapping. This should work inter-thread as well.
   int mode = bTry ? F_TLOCK : F_LOCK;

   if( lockf(d->filefd, mode, 0 ) != 0 )
   {
      if( bTry && (errno == EAGAIN || errno == EACCES))
      {
         // failed to acquire the lock.
         return false;
      }

      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                  .extra("lockf" )
                  .sysError( (int32) errno )
                  );
   }

   // synchronize the version
   //msync( d->bd, sizeof(BufferData), MS_SYNC );

   // resize the data, if necessary
   if( (d->mapsize < offset + size) || (bTrunc && d->bd->size != offset + size) )
   {
      // unmap, we need to remap the file.
      if( d->cache != MAP_FAILED )
      {
         if( munmap( d->cache, d->mapsize ) )
         {
            lockf(d->filefd, F_UNLCK, 0 );
            throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                                   .extra("munmap" )
                                   .sysError( (int32) errno )
                                   );
         }
      }

      // actually resize the underlying file
      if ( ftruncate( d->filefd, offset + size + sizeof(BufferData) ) != 0 )
      {
         lockf(d->filefd, F_UNLCK, 0 );
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                                   .extra( String("ftruncate to ").N(size).A( " bytes" ) )
                                   .sysError( (int32) errno ) );
      }

      d->bd->size = offset + size;
      d->cache = MAP_FAILED;
      d->mapsize = 0;
   }

   // map the rest of the file, if necessary
   if( d->cache == MAP_FAILED )
   {
      d->cache = mmap( 0, d->bd->size, PROT_READ | PROT_WRITE, MAP_SHARED, d->filefd, sizeof(BufferData) );

      if( d->cache == MAP_FAILED )
      {
         lockf(d->filefd, F_UNLCK, 0 );
         throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                        .extra("mmap" )
                        .sysError( (int32) errno )
                        );
      }

      d->mapsize = d->bd->size;
   }

   memcpy( static_cast<char*>(d->cache)+offset, data, size );
   if( msync( d->bd, sizeof(BufferData), bSync ? MS_SYNC : MS_ASYNC ) != 0 )
   {
      lockf(d->filefd, F_UNLCK, 0 );
      throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_WRITE,
                           .extra("msync" )
                           .sysError( (int32) errno )
                           );
   }

   lockf(d->filefd, F_UNLCK, 0 );
   return true;
}

bool SharedMem::write( const void* data, int64 size, int64 offset, bool bSync, bool bTrunc )
{
   return internal_write(data, size, offset, bSync, false, bTrunc );
}

bool SharedMem::tryWrite( const void* data, int64 size, int64 offset, bool bSync, bool bTrunc )
{
   return internal_write(data, size, offset, bSync, true, bTrunc );
}

int64 SharedMem::size() const
{
   return d->mapsize;
}

}

/* end of sharedmem_posix.cpp */
