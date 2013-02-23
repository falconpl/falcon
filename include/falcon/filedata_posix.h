/*
   FALCON - The Falcon Programming Language.
   FILE: filedata_posix.h

   Abstraction for system-specific file descriptor/handler (POSIX)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SYSFILEDATA_POSIX_H_
#define _FALCON_SYSFILEDATA_POSIX_H_

#include <falcon/setup.h>

namespace Falcon {
namespace Sys {

class FALCON_DYN_CLASS FileData {
public:
   int fdFile;
   bool m_nonBloking;

   FileData() {}

   FileData( int fd, bool nb=false ):
      fdFile( fd ),
      m_nonBloking(nb)
   {}

   /**
    * Moves the system file data to the destination.
    */
   void passOn( FileData& destination )
   {
      destination.fdFile = fdFile;
      destination.m_nonBloking = m_nonBloking;
      fdFile = -1;
   }

private:
   // disable implicit copy
   FileData( FileData& )
   {}
};

}
}

#endif /* _FALCON_SYS_FILEDATA_POSIX_H_ */

/* end of filedata_posix.h */
