/*
   FALCON - The Falcon Programming Language.
   FILE: filedata_win.h

   Abstraction for system-specific file descriptor/handler (WINDOWS)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_SYS_FILEDATA_WIN_H_
#define _FALCON_SYS_FILEDATA_WIN_H_

#include <falcon/setup.h>
#include <windows.h>

namespace Falcon {
namespace Sys {
class FileData {
public:
   HANDLE hFile;
   bool bIsFile;
   bool bNonBlocking;

   FileData( HANDLE hf = NULL, bool bf = true, bool nb = false ):
      hFile( hf ),
      bIsFile( bf ),
      bNonBlocking( nb )
   {}

   void passOn( FileData& destination )
   {
      destination.hFile = hFile;
      destination.bIsFile = bIsFile;
      destination.bNonBlocking = bNonBlocking;
      hFile = NULL;
   }

private:
   // disable implicit copy
   FileData( FileData& )
   {}
};

}
}

#endif /* _FALCON_SYSFILEDATA_WIN_H_ */

/* end of filedata_win.h */
