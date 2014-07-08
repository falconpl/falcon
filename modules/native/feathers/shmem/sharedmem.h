/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem.h

   Stream on a shared disk file.

   Interprocess shared-memory object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_FEATHERS_SHAREDMEM_H_
#define _FALCON_FEATHERS_SHAREDMEM_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/stream.h>

namespace Falcon {

/** Shared memory abstraction.
   This class allows a closed, black-box shared memory area to exchange data between
   Falcon applications.

   The data in the shared memory cannot be accessed directly by the applications. Instead,
   it must be read or written atomically to some process-specific memory, and then
   accessed from there. Even if generating extra reads and writes at each
   passage, this method avoids that programming errors in one application (which may
   be a script) block the others.

   The local memory where the shared memory must be copied is accessed in form
   of a Stream. The stream interface is extremely efficient for memory based operation,
   and has already all the needed policies for dynamic memory management, and read-write
   size recording.

   The shared memory can be backed on a system file or just be created transient.

 */
class SharedMem
{
public:
   /** Creates a non-open shared memory object. */
   SharedMem();

   /** Creates and opens a shared memory object that can be backed up on the given system file.
       @param Name name for the shared memory object, or filename if bFileBackup is true.
       @param bFileBackup true to store the shared memory object persistently on the filesystem.
       @throws Falcon::IoError if the memory object cannot be created, or if
       the file cannot be created.

       If the object has backup policy, and the given file exists when the
       object is created,  the lastVersion() of this object is updated so
       that the first read is known to be aligned.

       Backup policies are left to the underlying O/S.
    */
   SharedMem( const String &name, bool bFileBackup=false );

   /** Frees shared memory resources. */
   ~SharedMem();

   /** Initializes the object.
     @param Name name for the shared memory object, or filename if bFileBackup is true.
     @param bOpen if true, open a previously existing resource if present.
     @param bFileBackup true to store the shared memory object persistently on the filesystem.
     @throws Falcon::IoError if the memory object cannot be created, or if
     the file cannot be created.
     @throws Falcon::CodeError if the object is already initialized.

     If the object has backup policy, and the given file exists when the
     object is created,  the lastVersion() of this object is updated so
     that the first read is known to be aligned.

     If bOpen is true and the resource already exists, it is used as-is, otherwise
     it's reset and cleared (as if created on the spot).

     Backup policies are left to the underlying O/S.
    */
   void init( const String &name, bool bOpen = true, bool bFileBackup=false );

   /** Tries to open an existing resource or create it anew.

    */
   void open(const String &name, bool bFileBackup=false ) { init(name, true, bFileBackup); }

   /** Creates a new resource, eventually destroing it if existing.
    */
   void create(const String &name,  bool bFileBackup=false ) { init(name, false, bFileBackup); }
   int64 read( void* data, int64 size, int64 offset = 0 );
   bool grab(void* data, int64& size, int64 offset = 0 );
   void* grabAll( int64& size );
   bool write( const void* data, int64 size, int64 offset = 0, bool bSync = false, bool bTrunc = false );

   /** Closes the resource, eventually removing it.
    \param bRemove remove the resource from the system.
    */
   void close( bool bRemove = false );

   /** Returns the latest version that has been read or committed.

       Initially, the shared memory has version 1, and the local view
       of the memory has the same version.

       If the shared memory is not initialized,
       or if nothing has been written yet, the version is 0.
    */
   uint32 version() const;

   int64 size();
   int64 localSize() const;

private:
   class Private;
   Private* d;

   int64 lockAndAlign();
   void unlock();
   bool internal_write( const void* data, int64 size, int64 offset, bool bSync, bool bTrunc );
};

}

#endif /* _FALCON_SHAREDMEM_H_ */

/* end of sharedmem.h */
