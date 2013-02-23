/*
   FALCON - The Falcon Programming Language.
   FILE: filedata.h

   Abstraction for system-specific file descriptor/handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SYS_FILEDATA_H_
#define _FALCON_SYS_FILEDATA_H_

#include <falcon/setup.h>

namespace Falcon {
namespace Sys {
/** Base class for file stream system data.
 * This is an empty class that is inherited by the concrete
 * system-specific file stream data. It is used to provide
 * an abstract representation of the raw system file structures,
 * descriptors or pointers, and associated utility data.
 *
 * The concrete implementation is different depending
 * on the target final system (the final system implementation
 * just declares the class).
 */

class FileData;

}
}


#ifdef FALCON_SYSTEM_WIN
#include <falcon/filedata_win.h>
#else
#include <falcon/filedata_posix.h>
#endif


#endif /* _FALCON_SYS_FILEDATA_H_ */

/* end of filedata.h */
