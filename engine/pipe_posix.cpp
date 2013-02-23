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
   const SysFileData& writeSide() const { return m_writeSide; }

   /** Returns the side of the pipe that can be read. */
   const SysFileData& readSide() const { return m_readSide; }

   /** Returns the side of the pipe that can be written. */
   SysFileData& writeSide() { return m_writeSide; }
   /** Returns the side of the pipe that can be read. */
   SysFileData& readSide() { return m_readSide; }

   void closeRead();
   void closeWrite();

   void close();

private:
   FileData m_readSide;
   FileData m_writeSide;
};

}
}

#endif

/* end of pipe.h */
