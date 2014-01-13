/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: process_sys_win.cpp

   MS-Windows specific implementation of openProcess
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun Jan 30 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   MS-Windows specific implementation of openProcess
*/

#include <falcon/stream.h>
#include <falcon/vmcontext.h>
#include <falcon/pipestreams.h>

#include "process_mod.h"
#include "process_ext.h"
#include "process.h"

#include <tlhelp32.h>

namespace Falcon {
namespace Mod {

namespace {

struct AutoBuf
{
   TCHAR* buf;
   AutoBuf(String& source) :
      buf(0)
   {
      size_t bufSize = source.length() * 4 + 1;
      buf = new TCHAR[ bufSize ];
#ifdef UNICODE
      source.toWideString( buf, bufSize );
#else
      source.toCString( buf, bufSize );
#endif
   }

   ~AutoBuf()
   {
      if(buf) delete [] buf;
   }
};


static void s_fullCommand(const String& command, String& finalCmd)
{
   Falcon::length_t pos = command.find(' ');
   if( pos != String::npos )
   {
      finalCmd = command.subString(0, pos );
   }
   else
   {
      finalCmd = command;
   }

   AutoBuf fileNameBuf(finalCmd);
   TCHAR fullCommand[1024];
   TCHAR* filePart;
   if ( SearchPath( NULL, fileNameBuf.buf, NULL, 1024, fullCommand, &filePart ) > 0)
   {
      finalCmd.bufferize(fileNameBuf.buf) + command.subString(pos);
   }
   else
   {
      finalCmd = command;
   }
}

} // anonymous namespace


//====================================================================
// Simple process manipulation functions

uint64 processId()
{
   return (uint64) GetCurrentProcessId();
}

uint64 threadId()
{
   return (uint64) GetCurrentThreadId();
}

bool processKill( uint64 id )
{
   HANDLE hProc = OpenProcess( PROCESS_TERMINATE, FALSE, (uint32) id );
   if ( hProc )
   {
      CloseHandle( hProc );
      return true;
   }
   return false;
}

bool processTerminate( uint64 id )
{
   HANDLE hProc = OpenProcess( PROCESS_TERMINATE, FALSE, (uint32)id );
   if ( hProc )
   {
      CloseHandle( hProc );
      return true;
   }
   return false;
}

//====================================================================
// Process enumerator
//====================================================================

namespace {
class WIN_PROC_HANDLE
{
public:
   HANDLE hSnap;
   PROCESSENTRY32 process;
};
}


ProcessEnum::ProcessEnum()
{
   WIN_PROC_HANDLE *ph = new WIN_PROC_HANDLE;
   ph->hSnap = CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS, 0 );
   ph->process.dwSize = sizeof( ph->process );
   if ( Process32First( ph->hSnap, &ph->process ) )
   {
      m_sysdata = ph;
   }
   else
   {
      CloseHandle( ph->hSnap );
      delete ph;
      m_sysdata = 0;

       throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError, FALCON_PROCESS_ERROR_ERRLIST3,
               .desc(FALCON_PROCESS_ERROR_ERRLIST3_MSG )
               .sysError((uint32) GetLastError() ));
   }
}

ProcessEnum::~ProcessEnum()
{
   this->close();
}

bool ProcessEnum::next()
{
   if ( m_sysdata == 0 )
   {
      return false;
   }

   WIN_PROC_HANDLE *ph = reinterpret_cast<WIN_PROC_HANDLE*>( m_sysdata );
   m_pid = ph->process.th32ProcessID;
   m_ppid = ph->process.th32ParentProcessID;
   m_commandLine.bufferize( ph->process.szExeFile );
   m_name.bufferize( ph->process.szExeFile );

   if ( ! Process32Next( ph->hSnap, &ph->process ) )
   {
      CloseHandle( ph->hSnap );
      delete ph;
      m_sysdata = 0;
      DWORD dwLastError = GetLastError();
      if( dwLastError != ERROR_NO_MORE_FILES )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError, FALCON_PROCESS_ERROR_ERRLIST4,
                  .desc(FALCON_PROCESS_ERROR_ERRLIST4_MSG )
                  .sysError( dwLastError )
                  );
      }

      // no more files.
      return false;
   }

   return true;
}


void ProcessEnum::close()
{
   if ( ! m_sysdata  )
   {
      return;
   }

   WIN_PROC_HANDLE *ph = (WIN_PROC_HANDLE *) m_sysdata;
   if ( ! CloseHandle( ph->hSnap ) )
   {
      delete ph;
      m_sysdata = 0;
      throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
               FALCON_PROCESS_ERROR_ERRLIST2,
                 .desc(FALCON_PROCESS_ERROR_ERRLIST2_MSG )
                 .sysError( (uint32) GetLastError() )
                 );
   }
   delete ph;
   m_sysdata = 0;
}

//====================================================================
// Generic system interface.
//====================================================================


class Process::Private
{
public:
   HANDLE hPipeInRd;
   HANDLE hPipeInWr;
   HANDLE hPipeOutRd;
   HANDLE hPipeOutWr;
   HANDLE hPipeErrRd;
   HANDLE hPipeErrWr;

   HANDLE hChildProcess;
   DWORD hChildProcID;

   Private() {
      hChildProcess = INVALID_HANDLE_VALUE;

      hPipeInRd = INVALID_HANDLE_VALUE;
      hPipeInWr = INVALID_HANDLE_VALUE;
      hPipeOutRd = INVALID_HANDLE_VALUE;
      hPipeOutWr = INVALID_HANDLE_VALUE;
      hPipeErrRd = INVALID_HANDLE_VALUE;
      hPipeErrWr = INVALID_HANDLE_VALUE;
   }

   ~Private() {

   }

   void closeAll()
   {
      if( hPipeInRd != INVALID_HANDLE_VALUE ) { CloseHandle( hPipeInRd ); }
      if( hPipeInWr != INVALID_HANDLE_VALUE ) { CloseHandle( hPipeInWr ); }
      if( hPipeOutRd != INVALID_HANDLE_VALUE ) { CloseHandle( hPipeOutRd ); }
      if( hPipeOutWr != INVALID_HANDLE_VALUE ) { CloseHandle( hPipeOutWr ); }
      if( hPipeErrRd != INVALID_HANDLE_VALUE ) { CloseHandle( hPipeErrRd ); }
      if( hPipeErrWr != INVALID_HANDLE_VALUE ) { CloseHandle( hPipeErrWr ); }
   }
};


static const char *shellName()
{
   const char *shname = getenv("ComSpec");
   if ( shname == 0 ) {
      OSVERSIONINFO osVer;
      osVer.dwOSVersionInfoSize = sizeof( osVer );
      if( GetVersionEx( &osVer ) && osVer.dwPlatformId == VER_PLATFORM_WIN32_NT )
         shname = "CMD.EXE";
      else
         shname = "COMMAND.COM";
   }
   return shname;
}

static const char *shellParam()
{
   return "/C";
}

bool Process::terminate( bool )
{
   m_mtx.lock();
   if( (! m_bOpen) || m_done )
   {
      m_mtx.unlock();
      return false;
   }
   m_mtx.unlock();

   if( ! CloseHandle( _p->hChildProcess ) )
   {
      throw FALCON_SIGN_XERROR(::Falcon::Ext::ProcessError,
               FALCON_PROCESS_ERROR_TERMINATE,
               .desc(FALCON_PROCESS_ERROR_TERMINATE_MSG)
               .sysError((unsigned int) GetLastError() ) );
   }

   return true;
}

void Process::sys_init()
{
   _p = new Private();

}


void Process::sys_destroy()
{
   delete _p;
}


void Process::sys_wait()
{
   if( WaitForSingleObject( _p->hChildProcess, INFINITE ) == WAIT_FAILED )
   {
      throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError, FALCON_PROCESS_ERROR_WAITFAIL,
         .desc(FALCON_PROCESS_ERROR_WAITFAIL_MSG)
         .sysError( (uint32) GetLastError() ) );
   }

   m_exitval = -1;
   DWORD dwExitCode = 0;
   GetExitCodeProcess( _p->hChildProcess, &dwExitCode );

   if( dwExitCode )
   {
      m_exitval = (int) static_cast<unsigned int>(dwExitCode);
   }
}


void Process::sys_close()
{
   if( _p->hChildProcess != INVALID_HANDLE_VALUE )
   {
      CloseHandle(_p->hChildProcess);
      _p->hChildProcess = INVALID_HANDLE_VALUE;
   }
}


void Process::sys_open( const String& cmd, int params )
{
   // prepare security attributes
   SECURITY_ATTRIBUTES secAtt;
   secAtt.nLength = sizeof( secAtt );
   secAtt.lpSecurityDescriptor = NULL;
   secAtt.bInheritHandle = TRUE;

   try
   {
      if ( !CreatePipe( &_p->hPipeInRd, &_p->hPipeInWr, &secAtt, 0 ) )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
               FALCON_PROCESS_ERROR_OPEN_PIPE,
               .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
               .extra("IN pipe")
               .sysError(GetLastError()) );
      }

      if ( ! CreatePipe( &_p->hPipeOutRd, &_p->hPipeOutWr, &secAtt, 0 ) )
      {
         throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
               FALCON_PROCESS_ERROR_OPEN_PIPE,
               .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
               .extra("OUT pipe")
               .sysError(GetLastError()) );
      }

      if ( (params & MERGE_AUX) != 0 )
      {
         _p->hPipeErrRd = _p->hPipeOutRd;
         _p->hPipeErrWr = _p->hPipeOutWr;
      }
      else
      {
         if ( !CreatePipe( &_p->hPipeErrRd, &_p->hPipeErrWr, &secAtt, 0 ) )
         {
            throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
                  FALCON_PROCESS_ERROR_OPEN_PIPE,
                  .desc( FALCON_PROCESS_ERROR_OPEN_PIPE_MSG )
                  .extra("AUX pipe")
                  .sysError(GetLastError()) );
         }
      }
   }
   catch(...)
   {
      _p->closeAll();
      throw;
   }

   String fullCmd;
   if( (params & USE_PATH) != 0 )
   {
      s_fullCommand( cmd, fullCmd );
   }
   else {
      fullCmd = cmd;
   }

   if( (params & USE_SHELL) != 0 )
   {
      fullCmd = String(shellName()) + " " + shellParam() + fullCmd;
   }

   STARTUPINFO si;
   PROCESS_INFORMATION proc;
   DWORD iFlags = 0;
   memset( &si, 0, sizeof( si ) );
   si.cb = sizeof( si );

   si.dwFlags = STARTF_USESTDHANDLES;
   si.hStdInput = _p->hPipeInRd;
   si.hStdOutput = _p->hPipeOutWr;
   si.hStdError = _p->hPipeErrWr;

   if( (params & BACKGROUND) != 0 )
   {
      si.dwFlags |= STARTF_USESHOWWINDOW;
      si.wShowWindow = SW_HIDE;
      iFlags |= DETACHED_PROCESS;
   }

   AutoBuf cmdbuf(fullCmd);
   if ( ! CreateProcess( NULL,
                         cmdbuf.buf,
                         NULL,
                         NULL,
                         TRUE, //Inerhit handles!
                         iFlags,
                         NULL,
                         NULL,
                         &si,
                         &proc
                         ) )
   {
      _p->closeAll();

      throw FALCON_SIGN_XERROR( ::Falcon::Ext::ProcessError,
                  FALCON_PROCESS_ERROR_OPEN,
                  .desc( FALCON_PROCESS_ERROR_OPEN_MSG )
                  .sysError(GetLastError()) );
   }
   else
   {
     _p->hChildProcID = proc.dwProcessId;
     _p->hChildProcess = proc.hProcess;

     CloseHandle( proc.hThread ); // unused

      // close unused pipe ends
      CloseHandle( _p->hPipeInRd );
      CloseHandle( _p->hPipeOutWr );
      if( (params & MERGE_AUX) == 0 )
      {
         CloseHandle( _p->hPipeErrWr );
      }

      // save the system-specific file streams, if not sunk
      if ( (params & SINK_INPUT) == 0 )
      {
         m_stdIn = new WritePipeStream( new Sys::FileData(_p->hPipeInWr) );
      }
      else {
         CloseHandle( _p->hPipeInWr );
      }

      if ( (params & SINK_OUTPUT) == 0 )
      {
         m_stdOut = new ReadPipeStream( new Sys::FileData(_p->hPipeOutRd) );
      }
      else
      {
         CloseHandle( _p->hPipeOutRd );
      }

      if ( (params & (SINK_AUX|MERGE_AUX) ) == 0 )
      {
         m_stdErr = new ReadPipeStream( new Sys::FileData(_p->hPipeErrRd) );
      }
      else
      {
         CloseHandle( _p->hPipeErrRd );
      }
   }
}

int64 Process::pid() const
{
   return (int64) _p->hChildProcID;
}

}} // ns Falcon::Sys

/* end of process_sys_win.cpp */
