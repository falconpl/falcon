/*
   FALCON - The Falcon Programming Language.
   FILE: pipe.h

   System independent abstraction for linked inter-process sockets.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SYS_PIPE_H_
#define _FALCON_SYS_PIPE_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/filedata.h>
#include <falcon/multiplex.h>

#include <falcon/fstream.h>
#include <falcon/stdmpxfactories.h>

namespace Falcon {
namespace Sys {

/** System independent abstraction for linked inter-process sockets.
 *
 * This provides an abstraction layer presenting a system-specific
 * socket pair as a pair of SysStreamData.
 *
 * The system-specific implementation will then use that SysStreamData
 * as the implementation on that system requires.
 *
 * When the pipe is destroyed, its sockets are closed, unless
 * they have been moved elsewhere using SysFileData::passOn.
 *
 */

class FALCON_DYN_CLASS Pipe
{
public:
   /**
    * Creates the pipe.
    *
    * \throws IoError in case of system error.
    *
    */
   Pipe();
   ~Pipe();

   /** Returns the side of the pipe that can be written. */
   const FileData& writeSide() const { return *m_writeSide; }

   /** Returns the side of the pipe that can be read. */
   const FileData& readSide() const { return *m_readSide; }

   /** Returns the side of the pipe that can be written. */
   FileData& writeSide() { return *m_writeSide; }
   /** Returns the side of the pipe that can be read. */
   FileData& readSide() { return *m_readSide; }

   void closeRead();
   void closeWrite();

   void close();

   /** Create a read-only stream out of the read side of the pipe.
    * The read-side stream data is passed on the newly created read stream;
    * from that moment on, the system descriptor for the read side
    * won't be accessible through the readSide() method, and closing
    * this Pipe won't close the read side.
    */
   Stream* getReadStream();

   /** Create a write-only stream out of the write side of the pipe.
    * The write-side stream data is passed on the newly created read stream;
    * from that moment on, the system descriptor for the write side
    * won't be accessible through the writeSide() method, and closing
    * this Pipe won't close the write side.
    */
   Stream* getWriteStream();

   /** Traits for streams that can be interpreted as directional, piped FileData.
    */
   class FALCON_DYN_CLASS MpxFactory: public Multiplex::Factory
   {
   public:
      /**
       * Creates a trait instance.
       * \pram readDirection if true, this pipe traits are created for read direction.
       */
      MpxFactory( bool readDirection );
      virtual ~MpxFactory();

      virtual Multiplex* create( Selector* selector ) const;

      bool isRead() const { return m_bReadDirection; }

   private:
      class ReadMPX;
      class WriteMPX;

      bool m_bReadDirection;
   };

private:
   FileData* m_readSide;
   FileData* m_writeSide;
};

}
}

#endif

/* end of pipe.h */
