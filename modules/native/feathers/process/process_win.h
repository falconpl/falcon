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

#include "process.h"

   //class FileService;
namespace Falcon { namespace Sys {

class WinProcess: public Process
{
public:
   WinProcess();
   ~WinProcess();

   /*
    * Interface Implementation
    */
   Falcon::Stream *inputStream();
   Falcon::Stream *outputStream();
   Falcon::Stream *errorStream();
   //
   bool close();
   bool wait( bool block );
   bool terminate( bool severe = false );
   
   DWORD pid() const { return m_procId; }
   
private:   
   friend bool openProcess(Process* ph, String** argList, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg );

   HANDLE hPipeInRd;
   HANDLE hPipeInWr;
   HANDLE hPipeOutRd;
   HANDLE hPipeOutWr;
   HANDLE hPipeErrRd;
   HANDLE hPipeErrWr;

   HANDLE m_procHandle;
   DWORD m_procId;
};


typedef struct tag_winProcHandle {
   HANDLE hSnap;
   PROCESSENTRY32 process;
} WIN_PROC_HANDLE;


}} // ns Falcon::Sys

#endif

/* end of process_sys_win.h */
