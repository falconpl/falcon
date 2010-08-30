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

// To make SUNC happy
#define _POSIX_C_SOURCE 3

#include <falcon/wopi/sharedmem.h>
#include <falcon/autocstring.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>
#include <falcon/stream.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <semaphore.h>
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>

#define SEM_PREFIX "/WPS_"
#define APP_PREFIX "/WPM_"

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
      bSemReady(false),
      shmfd(0),
      filefd(0),
      bd(0)
      {}

   bool bSemReady;
   sem_t *sema;
   int shmfd;
   int filefd;

   // Memory mapped data
   BufferData* bd;
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
   String sSemName =  "WOPI_SEM_"+name;

   try
   {
      AutoCString csn( sSemName );

      // create the semaphore, with an initial value of 0
      // so it is initially takeable
      if( (d->sema = sem_open( csn.c_str(), O_CREAT, 0666, 1 )) == SEM_FAILED )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
               .extra( "sem_open " + sSemName )
               .sysError( errno ) );
      }

      d->bSemReady = true;
      //sem_unlink( csn.c_str() );


      // ok, get the semaphore.
      if ( sem_wait( d->sema ) != 0 )
      {

         throw new IoError( ErrorParam( e_io_error, __LINE__ )
                     .extra("sem_wait")
                     .sysError( errno ) );
      }

      // we're the owners of the memory. But, is it new or does it exists?
      int fd = 0; // the descriptor to map.
      if( filename != "" )
      {
         // try to map the file
         AutoCString cfname( filename );
         d->filefd = open( cfname.c_str(), O_CREAT | O_RDWR, 0666 );
         if( d->filefd <= 0 )
         {
            throw new IoError( ErrorParam( e_io_error, __LINE__ )
                        .extra("open "+ filename )
                        .sysError( errno ) );
         }

         fd = d->filefd;
      }
      else
      {
         String sMemName = APP_PREFIX + name;
         AutoCString cMemName( sMemName );
         d->shmfd = shm_open( cMemName.c_str(), O_CREAT | O_RDWR, 0666 );

         if( d->shmfd <= 0 )
         {
            throw new IoError( ErrorParam( e_io_error, __LINE__ )
                              .extra("shm_open "+ sMemName )
                              .sysError( errno ) );
         }

         //shm_unlink( cMemName.c_str() );
         fd = d->shmfd;
      }

      // eventually prepare the first buffer
      init();

      // ok, let's run.
      sem_post( d->sema );
   }
   catch( ... )
   {
      if( d->bSemReady )
      {
         sem_post( d->sema );
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
   int fd = d->shmfd <= 0 ? d->filefd : d->shmfd;

   // ok, we have our stream. See if it needs initialization.
   off_t pos = lseek( fd, 0, SEEK_END );
   if( pos < 0 )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                    .extra("lseek" )
                                    .sysError( errno ) );
   }


   // Yes? -- add space.
   if( pos == 0 )
   {
      if ( ftruncate( fd, sizeof(BufferData) ) != 0 )
      {
         throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                             .extra("ftruncate" )
                                             .sysError( errno ) );
      }
   }
   d->bd = (BufferData*) mmap( 0, sizeof(BufferData), PROT_READ | PROT_WRITE, MAP_SHARED,
         fd, 0 );

   if( d->bd == MAP_FAILED )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                                .extra("mmap" )
                                                .sysError( errno ) );
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
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                                     .extra("msync" )
                                                     .sysError( errno ) );
   }
}

void SharedMem::close()
{
   if( d->bd != MAP_FAILED )
   {
      munmap( d->bd, sizeof(BufferData) );
   }

   // we're in trouble.
   if( d->bSemReady )
   {
      sem_close( d->sema );
      d->bSemReady = false;
   }

   if( d->filefd > 0 )
   {
      ::close( d->filefd );
      d->filefd = 0;
   }

   if( d->shmfd > 0 )
   {
      ::close( d->shmfd );
      d->shmfd = 0;
   }
}


bool SharedMem::read( Stream* target, bool bAlwaysRead )
{
   // acquire adequate memory mapping.
   if( sem_wait( d->sema ) != 0 )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                         .extra("sem_wait" )
                                         .sysError( errno ) );
   }

   // be sure we have the right data in
   msync( d->bd, sizeof(BufferData), MS_SYNC );

   // are we aligned?
   if( m_version != d->bd->version && ! bAlwaysRead )
   {
      sem_post( d->sema );
      return false;
   }

   // align
   try
   {
      internal_read( target );
      sem_post( d->sema );
   }
   catch( ... )
   {
      sem_post( d->sema );
      throw;
   }

   return true;
}


bool SharedMem::commit( Stream* source, int32 size, bool bReread  )
{
   // acquire adequate memory mapping.
   if( sem_wait( d->sema ) != 0 )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                         .extra("sem_wait" )
                                         .sysError( errno ) );
   }

   // synchronize the version
   msync( d->bd, sizeof(BufferData), MS_SYNC );

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
            sem_post( d->sema );
            throw;
         }
      }

      sem_post( d->sema );
      return false;
   }

   // write the new data.
   int fd = d->shmfd <= 0 ? d->filefd : d->shmfd;
   if ( ftruncate( fd, size + sizeof(BufferData) ) != 0 )
   {
      sem_post( d->sema );
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                                .extra( String("ftruncate to ").N(size).A( " bytes" ) )
                                .sysError( errno ) );
   }

   // map the rest of the file
   void* data = mmap( 0, size + sizeof(BufferData), PROT_READ | PROT_WRITE, MAP_SHARED,
         fd, 0 );

   if( data == MAP_FAILED )
   {
      sem_post( d->sema );
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                              .extra( String("mmap ").N( (int32) size).A(" bytes") )
                              .sysError( errno ) );
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
                           .sysError( source->lastError() ) );
         }
      }

      // update the version
      m_version++;
      if( m_version == 0 )
         m_version = 1;

      d->bd->version = m_version;
      d->bd->size = size;

      // sync all the buffers, infos and data
      msync( d->bd, size + sizeof(BufferData), MS_ASYNC );
      munmap( data, size );
      sem_post( d->sema );
   }
   catch( ... )
   {
      munmap( data, size );
      sem_post( d->sema );
      throw;
   }

   return true;
}

void SharedMem::internal_read( Stream* target )
{
   m_version = d->bd->version;
   int32 size = d->bd->size;
   int fd = d->shmfd <= 0 ? d->filefd : d->shmfd;

   // map the rest of the file
   void* data = mmap( 0, size + sizeof(BufferData), PROT_READ | PROT_WRITE, MAP_SHARED,
         fd, 0 );

   if( data == MAP_FAILED )
   {
      throw new IoError( ErrorParam( e_io_error, __LINE__ )
                     .extra( String("mmap ").N( (int32) size).A(" bytes") )
                     .sysError( errno ) );
   }

   try
   {
      byte* bdata = ((byte*) data) + sizeof(BufferData);
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
                           .sysError( target->lastError() ) );
         }
      }
      munmap( data, size );
   }
   catch( ... )
   {
      munmap( data, size );
      throw;
   }
}

uint32 SharedMem::currentVersion() const
{
   msync( d->bd, sizeof(BufferData), MS_SYNC );
   return d->bd->version;
}

}

/* end of sharedmem_posix.cpp */
