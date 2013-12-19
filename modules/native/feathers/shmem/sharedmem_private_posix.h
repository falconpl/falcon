/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem_private_posix.cpp

   Interprocess shared-memory object -- POSIX specific private part
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "sharedmem.h"

namespace Falcon {

// data in the initial part of the buffer
typedef struct tag_BufferData
{
   uint32 version;
   size_t size;
   char data[0];
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
      bd(static_cast<BufferData*>(MAP_FAILED))
      {}

   String name;
   atomic_int hasBeenInit;
   int shmfd;
   int filefd;
   uint32 version;
   int64 mapsize;

   // Memory mapped data
   BufferData* bd;

   bool s_lockf( int etype )
   {
      if( mlock(bd, mapsize + sizeof(BufferData)) != 0 )
      {
         throw FALCON_SIGN_XERROR(ShmemError, etype,
                     .extra("mlock" )
                     .sysError( (int32) errno )
                     );

      }

      return true;
   }

   void s_unlockf( int etype )
   {
      if( munlock(bd, mapsize + sizeof(BufferData)) != 0 )
      {
         throw FALCON_SIGN_XERROR(ShmemError, etype,
                     .extra("munlock" )
                     .sysError( (int32) errno )
                     );

      }
   }


   int64 lockAndAlign()
   {
      // acquire adequate memory mapping. This should work inter-thread as well.
      s_lockf(FALCON_ERROR_SHMEM_IO_READ);

      // d->bd is mem-mapped; any change is shared.
      int64 rdSize = bd->size;

      if( mapsize != rdSize  )
      {
         // -- no? -- realign
         if(munmap( bd, mapsize + sizeof(BufferData) ) != 0)
         {
            d->s_unlockf(FALCON_ERROR_SHMEM_IO_READ );
            throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                          .extra("munmap" )
                          .sysError( (int32) errno )
                          );
         }

         // map the rest of the file -- possibly change the cache.
         bd = (BufferData*) mmap( 0, rdSize+sizeof(BufferData), PROT_READ | PROT_WRITE, MAP_SHARED, filefd, 0 );

         if( d->bd == MAP_FAILED )
         {
            s_unlockf(FALCON_ERROR_SHMEM_IO_READ );
            throw FALCON_SIGN_XERROR(ShmemError, FALCON_ERROR_SHMEM_IO_READ,
                           .extra("mmap" )
                           .sysError( (int32) errno )
                           );
         }
         rdSize = mapsize = bd->size;
      }

      version = bd->version;

      return rdSize;
   }
};

}

#endif

/* end of sharedmem_private_posix.h */
