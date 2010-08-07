/*
   FALCON - The Falcon Programming Language.
   FILE: process_sys_win.h

   MS-Windows implementation of process handle
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   MS-Windows implementation of process handle
*/

#ifndef flc_process_sys_win_H
#define flc_process_sys_win_H

#include <windows.h>
#include <tlhelp32.h>

#include "process_sys.h"

namespace Falcon {

class FileService;

namespace Sys {

class WinProcessHandle: public ProcessHandle
{
   friend ProcessHandle *openProcess( String **argv, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg );

   HANDLE hPipeInRd;
   HANDLE hPipeInWr;
   HANDLE hPipeOutRd;
   HANDLE hPipeOutWr;
   HANDLE hPipeErrRd;
   HANDLE hPipeErrWr;

   HANDLE m_procHandle;
   DWORD m_procId;

public:
   WinProcessHandle():
      ProcessHandle()
   {}

   virtual ~WinProcessHandle();

   DWORD pid() const { return m_procId; }

   virtual ::Falcon::Stream *getInputStream();
   virtual ::Falcon::Stream *getOutputStream();
   virtual ::Falcon::Stream *getErrorStream();

   virtual bool close();
   virtual bool wait( bool block );
   virtual bool terminate( bool severe = false );
};

typedef struct tag_winProcHandle {
   HANDLE hSnap;
   PROCESSENTRY32 procent;
} WIN_PROC_HANDLE;

}
}

#endif

/* end of process_sys_win.h */
