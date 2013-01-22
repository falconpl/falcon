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

#include <falcon/fstream_sys_win.h>
#include <falcon/memory.h>

#include "process_win.h"

namespace Falcon { namespace Sys {


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


String s_fullCommand(String& command)
{
   String finalCmd;
   
   AutoBuf fileNameBuf(command);
   TCHAR fullCommand[1024];
   TCHAR* filePart;
   if ( ! SearchPath( NULL, fileNameBuf.buf, NULL, 1024, fullCommand, &filePart ) )
      finalCmd.bufferize(fileNameBuf.buf);
   else
      finalCmd.bufferize(fullCommand);

   return finalCmd;
}

} // anonymous namespace


//====================================================================
// Simple process manipulation functions

uint64 processId()
{
   return (uint64) GetCurrentProcessId();
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
ProcessEnum::ProcessEnum()
{
   WIN_PROC_HANDLE *ph = new WIN_PROC_HANDLE;
   ph->hSnap = CreateToolhelp32Snapshot( TH32CS_SNAPPROCESS, 0 );
   ph->process.dwSize = sizeof( ph->process );
   if ( Process32First( ph->hSnap, &ph->process ) )
      m_sysdata = ph;
   else
   {
      CloseHandle( ph->hSnap );
      delete ph;
      m_sysdata = 0;
   }
}

ProcessEnum::~ProcessEnum()
{
   this->close();
}

int ProcessEnum::next( String &name, uint64 &pid, uint64 &ppid, String &path )
{
   if ( m_sysdata == 0 )
      return 0;
   
   WIN_PROC_HANDLE *ph = reinterpret_cast<WIN_PROC_HANDLE*>( m_sysdata );
   pid = ph->process.th32ProcessID;
   ppid = ph->process.th32ParentProcessID;
   path.bufferize( ph->process.szExeFile );
   name.bufferize( ph->process.szExeFile );
   
   if ( ! Process32Next( ph->hSnap, &ph->process ) )
   {
      CloseHandle( ph->hSnap );
      delete ph;
      m_sysdata = 0;
   }
   
   return 1;
}

bool ProcessEnum::close()
{
   if ( ! m_sysdata  )
      return true;

   WIN_PROC_HANDLE *ph = (WIN_PROC_HANDLE *) m_sysdata;
   if ( ! CloseHandle( ph->hSnap ) )
   {
      delete ph;
      m_sysdata = 0;
      return false;
   }
   delete ph;
   m_sysdata = 0;

   return true;
}

//====================================================================
// Generic system interface.

bool spawn(String** argv, bool overlay, bool background, int *returnValue )
{
   String finalCmd = s_fullCommand(*argv[0]);
   
   // build the complete string
   for(size_t i = 1; argv[ i ] != 0; i++)
   {
      finalCmd.append( ' ' );
      finalCmd.append( *argv[i] );
   }

   PROCESS_INFORMATION proc;
   STARTUPINFO si;             
   memset( &si, 0, sizeof( si ) );
   si.cb = sizeof( si );
   
   DWORD iFlags = 0;      
   if( background )
   {
      iFlags = DETACHED_PROCESS;
      si.dwFlags = STARTF_USESHOWWINDOW;
      si.wShowWindow = SW_HIDE;
   }
      
   if( overlay )
   {
      si.hStdInput = GetStdHandle( STD_INPUT_HANDLE );
      si.hStdOutput = GetStdHandle( STD_OUTPUT_HANDLE );
      si.hStdError = GetStdHandle( STD_ERROR_HANDLE );
      si.dwFlags |= STARTF_USESTDHANDLES;
   }
   
   AutoBuf cmdbuf(finalCmd);
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
      *returnValue = GetLastError();
      return false;
   }
      
   // we have to change our streams with the ones of the process.
   WaitForSingleObject( proc.hProcess, INFINITE );
   DWORD iRet;
   GetExitCodeProcess( proc.hProcess, &iRet );
   //free( completeCommand );
   if ( overlay )
      _exit(iRet);
   
   CloseHandle( proc.hProcess );
   CloseHandle( proc.hThread );
   *returnValue = iRet;
   return (int) true;
}


bool spawn_read( String **argv, bool overlay, bool background, int *returnValue, String *sOut )
{
   String finalCmd = s_fullCommand(*argv[0]);
   
   // build the complete string
   for(size_t i = 1; argv[ i ] != 0; i++)
   {
      finalCmd.append( ' ' );
      finalCmd.append( *argv[i] );
   }
   HANDLE hRead = INVALID_HANDLE_VALUE;
   HANDLE hWrite = INVALID_HANDLE_VALUE;
   
   SECURITY_ATTRIBUTES secatt;
   secatt.nLength = sizeof( secatt );
   secatt.lpSecurityDescriptor = NULL;
   secatt.bInheritHandle = TRUE;
   
   if ( !CreatePipe( &hRead, &hWrite, &secatt, 0 ) )
   {
      *returnValue = GetLastError();
      return false;
   }
   
   PROCESS_INFORMATION proc;
   STARTUPINFO si;
   memset( &si, 0, sizeof( si ) );
   si.cb = sizeof( si );
   
   DWORD iFlags = 0;
   if( background )
   {
      iFlags = DETACHED_PROCESS;
      si.dwFlags = STARTF_USESHOWWINDOW; //| //STARTF_USESTDHANDLES
      si.wShowWindow = SW_HIDE;
   }
   
   if( overlay )
   {
      si.hStdInput = GetStdHandle( STD_INPUT_HANDLE );
      si.hStdError = GetStdHandle( STD_ERROR_HANDLE );
   }
      
   si.dwFlags |= STARTF_USESTDHANDLES;
   si.hStdOutput = hWrite;
   
   AutoBuf cmdbuf(finalCmd);
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
      *returnValue = GetLastError();
      return false;
   }   
   
   // read from the input handle
   char buffer[4096];
   DWORD readin;
   BOOL peek;
   bool signaled;
   
   do
   {
      signaled = WaitForSingleObject( proc.hProcess, 10 ) == WAIT_OBJECT_0;
      
      // nothing to read?
      peek = PeekNamedPipe( hRead, NULL, 0,  NULL, &readin, NULL );
      
      if ( readin > 0 )
      {
         ReadFile( hRead, buffer, readin, &readin, false );
         if ( readin != 0 )
         {
            String temp;
            temp.adopt( buffer, readin, 0 );
            sOut->append( temp );
         }
      }
      
   }
   while( readin > 0 || ! signaled );
   
   CloseHandle( hRead );
   DWORD iRet;
   // we have to change our streams with the ones of the process.
   GetExitCodeProcess( proc.hProcess, &iRet );
   //free( completeCommand );
   if ( overlay )
   {
      _exit(iRet);
   }
   
   CloseHandle( proc.hProcess );
   CloseHandle( proc.hThread );
   *returnValue = iRet;
   return (int) true;
}


const char *shellName()
{
   char *shname = getenv("ComSpec");
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

const char *shellParam()
{
   return "/C";
}

bool openProcess(Process* _ph, String **argv, bool sinkIn, bool sinkOut, bool sinkErr, bool mergeErr, bool bg )
{
   WinProcess* ph = static_cast<WinProcess*>(_ph);
   
   ph->hPipeInRd = INVALID_HANDLE_VALUE;
   ph->hPipeInWr = INVALID_HANDLE_VALUE;
   ph->hPipeOutRd = INVALID_HANDLE_VALUE;
   ph->hPipeOutWr = INVALID_HANDLE_VALUE;
   ph->hPipeErrRd = INVALID_HANDLE_VALUE;
   ph->hPipeErrWr = INVALID_HANDLE_VALUE;
   
   // prepare security attributes
   SECURITY_ATTRIBUTES secAtt;
   secAtt.nLength = sizeof( secAtt );
   secAtt.lpSecurityDescriptor = NULL;
   secAtt.bInheritHandle = TRUE;
   
   if ( ! sinkIn )
      if ( !CreatePipe( &ph->hPipeInRd, &ph->hPipeInWr, &secAtt, 0 ) )
      {
         ph->lastError( GetLastError() );
         return false;
      }
   
   if ( ! sinkOut )
   {
      if ( ! CreatePipe( &ph->hPipeOutRd, &ph->hPipeOutWr, &secAtt, 0 ) )
      {
         ph->lastError( GetLastError() );
         CloseHandle( ph->hPipeInRd );
         CloseHandle( ph->hPipeInWr );
         return false;
      }
      
      if ( mergeErr )
      {
         ph->hPipeErrRd = ph->hPipeOutRd;
         ph->hPipeErrWr = ph->hPipeOutWr;
      }
   }
   
   if ( ! sinkErr && ! mergeErr )
      if ( !CreatePipe( &ph->hPipeErrRd, &ph->hPipeErrWr, &secAtt, 0 ) )
      {
         ph->lastError( GetLastError() );
         CloseHandle( ph->hPipeInRd );
         CloseHandle( ph->hPipeInWr );
         CloseHandle( ph->hPipeOutRd );
         CloseHandle( ph->hPipeOutWr );
         return false;
      }
   
   String finalCmd = s_fullCommand(*argv[0]);
   // build the complete string
   for(size_t i = 1; argv[ i ] != 0; i++)
   {
      finalCmd.append( ' ' );
      finalCmd.append( *argv[i] );
   }

   STARTUPINFO si;
   PROCESS_INFORMATION proc;
   DWORD iFlags = 0;
   memset( &si, 0, sizeof( si ) );
   si.cb = sizeof( si );
   
   if ( ! bg )
   {
      // using show_hide AND using invalid handlers for unused streams
      si.dwFlags = STARTF_USESTDHANDLES;
      
      si.hStdInput = ph->hPipeInRd;
      si.hStdOutput = ph->hPipeOutWr;
      si.hStdError = ph->hPipeErrWr;
   }
   else
   {
      si.dwFlags |= STARTF_USESHOWWINDOW;
      si.wShowWindow = SW_HIDE;
      iFlags |= DETACHED_PROCESS;
   }

   AutoBuf cmdbuf(finalCmd);
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
      ph->lastError( GetLastError() );
      CloseHandle( ph->hPipeInWr );
      CloseHandle( ph->hPipeOutRd );
      CloseHandle( ph->hPipeErrRd );
   }
   else
   {
     ph->m_procId = proc.dwProcessId;
     ph->m_procHandle = proc.hProcess;
     
     CloseHandle( proc.hThread ); // unused
   }
   
   
   return true;
}

//====================================================================
// WinProcess system area.

WinProcess::WinProcess() :
   Process()
{ }

WinProcess::~WinProcess()
{
   if ( ! done() )
   {
      close();
      terminate( true );
      wait( true );
   }
}


bool WinProcess::wait( bool block )
{
   DWORD dw;

   if ( block ) {
      dw = WaitForSingleObject( m_procHandle, INFINITE );
   }
   else {
      dw = WaitForSingleObject( m_procHandle, 0 );
   }

   if ( dw == WAIT_OBJECT_0 ) {
      done( true );
      GetExitCodeProcess( m_procHandle, &dw );
      processValue( dw );
      CloseHandle( m_procHandle );
      return true;
   }
   else if ( dw == WAIT_TIMEOUT ) {
      done( false );
      return true;
   }

   lastError( GetLastError() );
   return false;
}

bool WinProcess::close()
{
   if ( hPipeInWr != INVALID_HANDLE_VALUE )
      CloseHandle( hPipeInWr );
   if ( hPipeOutRd != INVALID_HANDLE_VALUE )
      CloseHandle( hPipeOutRd  );
   if ( hPipeErrRd != INVALID_HANDLE_VALUE  )
      CloseHandle( hPipeErrRd );
   return true;
}

bool WinProcess::terminate( bool )
{
   if( TerminateProcess( m_procHandle, 0 ) ) {
      done( true );
      return true;
   }

   lastError( GetLastError() );
   return false;
}

::Falcon::Stream *WinProcess::inputStream()
{
   if( hPipeInWr == INVALID_HANDLE_VALUE || done() )
      return 0;

   WinFileSysData *data = new WinFileSysData( hPipeInWr, 0, false, WinFileSysData::e_dirOut, true );
   return new FileStream( data );
}

::Falcon::Stream *WinProcess::outputStream()
{
   if( hPipeOutRd == INVALID_HANDLE_VALUE || done() )
      return 0;

   WinFileSysData *data = new WinFileSysData( hPipeOutRd, 0, false, WinFileSysData::e_dirIn, true );

   return new FileStream( data );
}

::Falcon::Stream *WinProcess::errorStream()
{
   if( hPipeErrRd == INVALID_HANDLE_VALUE || done() )
      return 0;

   WinFileSysData *data = new WinFileSysData( hPipeErrRd, 0, false, WinFileSysData::e_dirIn, true );

   return new FileStream( data );

}

Process* Process::factory()
{
   return new WinProcess();
}

}} // ns Falcon::Sys

/* end of process_sys_win.cpp */
