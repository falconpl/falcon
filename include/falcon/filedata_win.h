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

namespace Falcon 
{
class Stream;

namespace Sys {

class FileData {
public:
   HANDLE hFile;
   bool bIsDiskFile;

   FileData( HANDLE hf = NULL, bool bf = true ):
      hFile( hf ),
      bIsDiskFile( bf )
   {      
   }

   void passOn( FileData& destination )
   {
      destination.hFile = hFile;
      destination.bIsDiskFile = bIsDiskFile;
      hFile = NULL;
   }

   virtual ~FileData() {}

private:
   // disable implicit copy
   FileData( FileData& )
   {}
};

class FileDataEx: public FileData 
{
public:
   bool bConsole;
   bool bBusy;

   typedef struct
   {
      OVERLAPPED overlapped;
      FileDataEx* self;
      Stream* owner;
      void* extra;
   }
   OVERLAPPED_EX;

   OVERLAPPED_EX ovl;

   HANDLE hEmulWrite;
   HANDLE hRealConsole;

   FileDataEx( HANDLE hf = NULL, bool bf = false, bool cons = false ):
      FileData(hf, bf),
      bConsole(cons),
      bBusy( false )
   {
      memset( &ovl, 0, sizeof(ovl) );
      ovl.self = this;
      hEmulWrite = INVALID_HANDLE_VALUE;
      hRealConsole = INVALID_HANDLE_VALUE;
   }

   virtual ~FileDataEx()
   {
      if(hEmulWrite != INVALID_HANDLE_VALUE) {
         CloseHandle(hEmulWrite);
      }
      if(hRealConsole != INVALID_HANDLE_VALUE) {
         CloseHandle(hRealConsole);
      }
   }
};

}
}

#endif /* _FALCON_SYSFILEDATA_WIN_H_ */

/* end of filedata_win.h */
