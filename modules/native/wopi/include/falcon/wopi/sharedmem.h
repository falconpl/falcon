/*
   FALCON - The Falcon Programming Language.
   FILE: sharedmem.h

   Stream on a shared disk file.

   TODO: Move this file in the main engine in the next version.

   Interprocess shared-memory object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Apr 2010 12:12:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SHAREDMEM_H_
#define _FALCON_SHAREDMEM_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>
#include <falcon/string.h>

namespace Falcon {

class SharedMemPrivate;

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

   @TODO write about the read-commit process.
 */
class SharedMem: public BaseAlloc
{
public:
   /** Creates a non-backed up shared memory object.
      @throws Falcon::IoError if the memory object cannot be created.
   */
   SharedMem( const String &name );

   /** Creates a shared memory object backed up on the given system file.
       @throws Falcon::IoError if the memory object cannot be created, or if
       the file cannot be created.

       If the file exists when the object is created, the lastVersion()
       of this object is updated so that the first read is known to be aligned.

       Backup policies are left to the underlying O/S.
    */
   SharedMem( const String &name, const String &filename );

   /** Frees shared memory resources. */
   ~SharedMem();

   /** Reads the content of the shared memory in the stream.

     If the stream is a memory based stream, this translates quite directly
     in a raw memcopy, with correct allocation strategies put in place for you.

     You can also use this method to write the contents of the shared memory in
     a different kind of stream, but be aware that the shared memory is held (read-lock)
     for the whole time needed for the copy. So, it's often preferable to read in a
     memory based stream and then transfer its content to a file.

     The content is read starting from the current position in the stream, and the
     file pointer is left past the full read position. You may want to truncate the
     file after the read is complete.

     The read will refresh the version reported by lastVersion(). If it is detected
     that the version didn't change, read returns false and nothing is done, unless
     bAlwaysRead is true. In that case, the read is performed even if the current
     version of the memory is detected to be the same as in the previous read.

      \param target The stream where to read the memory data.
      \param bAlwaysRead set to true to read the data even if the memory is still aligned.
      \return false if the data was not read (because of a mis-alignment in versions).
   */
   bool read( Stream* target, bool bAlwaysRead = false );

   /** Writes the contents of the target stream on the shared memory.

     If the stream is a memory based stream, this translates quite directly
     in a raw memcopy, with correct allocation strategies put in place for you.

     The commit takes into account the lastVersion() value. If the current version
     in the shared memory has been changed, the function returns false. If the bReread
     parameter is true, the target stream is read atomically, and lastVersion() is
     updated to the current version. To prevent the contents of the stream to be
     overwritten, pass bReread = false.

     In case of successful commit, the current version of the shared memory and the
     lastVersion() reported by this object are updated.

     \param source The stream where the source for the storage is located.
     \param bReread if true (default), will overwrite the contents of source.
     \param size The size of the data to be commited.
     \return true if the commit was possible, false if the version was changed in the
     meanwhile.
   */
   bool commit( Stream* source, int32 size, bool bReread = true );

   /** Return the current version of the memory.

       This method is meant to minimize operations in case of mis-alignments.

       It reads the current version of the shared memory. In case it is not
       the same as lastVersion(), the caller can safely assume that a commit()
       would fail, and avoid prepare the data to be stored.

       If it is the same as lastVersion(), then a commit MAY succeed, but it may still
       fail, so the caller need tobe prepared to this eventuality.
   */
   uint32 currentVersion() const;

   /** Returns the latest version that has been read or committed.
       Initially, the shared memory has version 1, and the local view
       of the memory has the same version.
    */
   uint32 lastVersion() const { return m_version; }


private:
   // Private D-pointer
   SharedMemPrivate* d;

   uint32 m_version;

   void internal_build( const String &name, const String &filename );
   void init();
   void close();
   void internal_read( Stream* target );
};

}

#endif /* _FALCON_SHAREDMEM_H_ */

/* end of sharedmem.h */
