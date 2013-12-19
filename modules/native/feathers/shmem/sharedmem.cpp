/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem.cpp

   Shared memory mapped object.

   Interprocess shared-memory object -- system indept. part.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/feathers/shmem/sharedmem.cpp"

#include <falcon/autocstring.h>
#include <falcon/stderrors.h>
#include <falcon/stream.h>
#include <falcon/atomic.h>

#include <string.h>

#include "sharedmem.h"
#include "errors.h"

#ifdef FALCON_SYSTEM_WIN
#include "sharedmem_private_win.h"
#else
#include "sharedmem_private_posix.h"
#endif


namespace Falcon {


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


int64 SharedMem::read( void* data, int64 size, int64 offset )
{
   // are we aligned?
   int64 rdSize = lockAndAlign();

   if ( offset >= rdSize )
   {
      size = 0;
   }
   else
   {
      // according to POSIX, d->bd is directly visible in all the processes.
      if( offset + size > rdSize )
      {
         size = rdSize - offset;
      }

      memcpy( data, reinterpret_cast<char*>(d->bd->data)+ offset, (size_t) size );
   }

   // release the file lock
   unlock();

   return size;
}



bool SharedMem::grab( void* data, int64& size, int64 offset )
{
   // acquire adequate memory mapping. This should work inter-thread as well.
   int64 rdSize = lockAndAlign();

   if( rdSize <= offset )
   {
      size = 0;
      unlock();
      return false;
   }

   if( size == 0 || rdSize < size+offset )
   {
      // we know rdSize is > offset because of the first check.
      size = rdSize - offset;
      unlock();
      return false;
   }

   memcpy( data, reinterpret_cast<char*>(d->bd->data) + offset, (size_t) size );

   // release the file lock
   unlock();

   return true;
}


void* SharedMem::grabAll( int64& size )
{
   // acquire adequate memory mapping. This should work inter-thread as well.
   int64 rdSize = lockAndAlign();

   void* data = malloc( (size_t) rdSize );
   if( data == 0 )
   {
      unlock();
      throw FALCON_SIGN_XERROR(CodeError, e_membuf_def, .extra(String("malloc(").N(rdSize).A(")") ));
   }

   size = rdSize;
   memcpy( data, reinterpret_cast<char*>(d->bd->data), (size_t) size );

   // release the file lock
   unlock();
   return data;
}


bool SharedMem::write( const void* data, int64 size, int64 offset, bool bSync, bool bTrunc )
{
   return internal_write(data, size, offset, bSync, bTrunc );
}

int64 SharedMem::size()
{
   int64 s = lockAndAlign();
   unlock();
   return s;
}

}

/* end of sharedmem.cpp */
