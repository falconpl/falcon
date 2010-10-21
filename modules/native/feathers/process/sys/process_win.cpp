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
   char* cbuf;
   wchar_t* wcbuf;
   AutoBuf(String& source, bool wide) :
      cbuf(0),
      wcbuf(0)
   {
      size_t bufSize = source.length() * 4 + 1;
      if(wide)
      {
         wcbuf = new wchar_t[ bufSize ];
         source.toWideString( wcbuf, bufSize );
      }
      else
	  {
         cbuf = new char[ bufSize ];
         source.toCString( cbuf, bufSize );
      }
      
   }

   ~AutoBuf()
   {
      if(cbuf) delete [] cbuf;
      if(wcbuf) delete [] wcbuf;
   }
};

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
   if ( CloseHandle( ph->hSnap ) )
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
   STARTUPINFOA si;
   STARTUPINFOW siw;
   PROCESS_INFORMATION proc;
   int iPos;
   DWORD iRet;
   DWORD iFlags;
   char fullCommand[2048];
   wchar_t *fullCommand_w = (wchar_t *)fullCommand;
   char *filePart;
   wchar_t *filePart_w;
   bool wideImplemented;
   String finalCmd;
   
   // find the command in the path
   uint32 bufSize = argv[0]->length() * 4 + 1;
   wchar_t *fileNameBuf = (wchar_t *) memAlloc( bufSize );
   argv[0]->toWideString( fileNameBuf, bufSize );
   
   if ( ! SearchPathW( NULL, fileNameBuf, NULL, 1024, fullCommand_w, &filePart_w ) )
   {
      if ( GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
      {
         wideImplemented = false;
         char *charbuf = (char *) fileNameBuf;
         if( argv[0]->toCString( charbuf, bufSize ) > 0 )
         {
            if ( ! SearchPathA( NULL, charbuf, NULL, 2048, fullCommand, &filePart ) )
               finalCmd = charbuf;
            else
               finalCmd = fullCommand;
         }
      }
      else {
         wideImplemented = true;
         finalCmd = fileNameBuf;
      }
   }
   else
      finalCmd = fullCommand_w;
   
   memFree( fileNameBuf );
   
   // build the complete string
   iPos = 1;
   while( argv[ iPos ] != 0 )
   {
      finalCmd.append( ' ' );
      finalCmd.append( *argv[iPos] );
      iPos++;
   }
   
   if( wideImplemented )
   {
      memset( &siw, 0, sizeof( siw ) );
      siw.cb = sizeof( siw );
      
      if( background )  {
         iFlags = DETACHED_PROCESS;
         siw.dwFlags = STARTF_USESHOWWINDOW; //| //STARTF_USESTDHANDLES
         siw.wShowWindow = SW_HIDE;
      }
      iFlags = 0;
      
      if( overlay ) {
         siw.hStdInput = GetStdHandle( STD_INPUT_HANDLE );
         siw.hStdOutput = GetStdHandle( STD_OUTPUT_HANDLE );
         siw.hStdError = GetStdHandle( STD_ERROR_HANDLE );
         siw.dwFlags |= STARTF_USESTDHANDLES;
      }
      
      bufSize = finalCmd.length() * 4 + 1;
      fileNameBuf = (wchar_t *) memAlloc( bufSize );
      finalCmd.toWideString( fileNameBuf, bufSize );
      if ( ! CreateProcessW( NULL,
                             fileNameBuf,
                             NULL,
                             NULL,
                             TRUE, //Inerhit handles!
                             iFlags,
                             NULL,
                             NULL,
                             &siw,
                             &proc
                             ) )
      {
         memFree( fileNameBuf );
         *returnValue = GetLastError();
         return false;
      }
      
      memFree( fileNameBuf );
   }
   else {
      memset( &si, 0, sizeof( si ) );
      si.cb = sizeof( si );
      
      if( background )  {
         iFlags = DETACHED_PROCESS;
         si.dwFlags = STARTF_USESHOWWINDOW; //| //STARTF_USESTDHANDLES
         si.wShowWindow = SW_HIDE;
      }
      iFlags = 0;
      
      if( overlay ) {
         si.hStdInput = GetStdHandle( STD_INPUT_HANDLE );
         si.hStdOutput = GetStdHandle( STD_OUTPUT_HANDLE );
         si.hStdError = GetStdHandle( STD_ERROR_HANDLE );
         si.dwFlags |= STARTF_USESTDHANDLES;
      }
      
      bufSize = finalCmd.length() * 4 + 1;
      char *charbuf = (char *) memAlloc( bufSize );
      finalCmd.toCString( charbuf, bufSize );
      
      if ( ! CreateProcessA( NULL,
                             charbuf,
                             NULL,
                             NULL,
                             TRUE,  // INHERIT HANDLES!
                             iFlags,
                             NULL,
                             NULL,
                             &si,
                             &proc
                             ) )
      {
         memFree( charbuf );
         *returnValue = GetLastError();
         return false;
      }
      
      memFree( charbuf );
   }
   
   // we have to change our streams with the ones of the process.
   WaitForSingleObject( proc.hProcess, INFINITE );
   GetExitCodeProcess( proc.hProcess, &iRet );
   //memFree( completeCommand );
   if ( overlay ) {
      _exit(iRet);
   }
   
   CloseHandle( proc.hProcess );
   CloseHandle( proc.hThread );
   *returnValue = iRet;
   return (int) true;
}


bool spawn_read( String **argv, bool overlay, bool background, int *returnValue, String *sOut )
{
   STARTUPINFOA si;
   STARTUPINFOW siw;
   PROCESS_INFORMATION proc;
   int iPos;
   DWORD iRet;
   DWORD iFlags;
   char fullCommand[2048];
   wchar_t *fullCommand_w = (wchar_t *)fullCommand;
   char *filePart;
   wchar_t *filePart_w;
   bool wideImplemented;
   String finalCmd;
   
   // find the command in the path
   uint32 bufSize = argv[0]->length() * 4 + 1;
   wchar_t *fileNameBuf = (wchar_t *) memAlloc( bufSize );
   argv[0]->toWideString( fileNameBuf, bufSize );
   
   if ( ! SearchPathW( NULL, fileNameBuf, NULL, 1024, fullCommand_w, &filePart_w ) )
   {
      if ( GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
      {
         wideImplemented = false;
         char *charbuf = (char *) fileNameBuf;
         if( argv[0]->toCString( charbuf, bufSize ) > 0 )
         {
            if ( ! SearchPathA( NULL, charbuf, NULL, 2048, fullCommand, &filePart ) )
               finalCmd = charbuf;
            else
               finalCmd = fullCommand;
         }
      }
      else {
         wideImplemented = true;
         finalCmd = fileNameBuf;
      }
   }
   else
      finalCmd = fullCommand_w;
   
   memFree( fileNameBuf );
   
   // build the complete string
   iPos = 1;
   while( argv[ iPos ] != 0 ) {
      finalCmd.append( ' ' );
      finalCmd.append( *argv[iPos] );
      iPos++;
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
   
   if( wideImplemented )
   {
      memset( &siw, 0, sizeof( siw ) );
      siw.cb = sizeof( siw );
      
      if( background )  {
         iFlags = DETACHED_PROCESS;
         siw.dwFlags = STARTF_USESHOWWINDOW; //| //STARTF_USESTDHANDLES
         siw.wShowWindow = SW_HIDE;
      }
      iFlags = 0;
      
      siw.hStdOutput = hWrite ;
      
      if( overlay ) {
         siw.hStdInput = GetStdHandle( STD_INPUT_HANDLE );
         siw.hStdError = GetStdHandle( STD_ERROR_HANDLE );
      }
      
      siw.dwFlags |= STARTF_USESTDHANDLES;
      bufSize = finalCmd.length() * 4 + 1;
      fileNameBuf = (wchar_t *) memAlloc( bufSize );
      finalCmd.toWideString( fileNameBuf, bufSize );
      if ( ! CreateProcessW( NULL,
                             fileNameBuf,
                             NULL,
                             NULL,
                             TRUE, //Inerhit handles!
                             iFlags,
                             NULL,
                             NULL,
                             &siw,
                             &proc
                             ) )
      {
         memFree( fileNameBuf );
         *returnValue = GetLastError();
         return false;
      }
      
      memFree( fileNameBuf );
   }
   else {
      memset( &si, 0, sizeof( si ) );
      si.cb = sizeof( si );
      
      if( background )  {
         iFlags = DETACHED_PROCESS;
         si.dwFlags = STARTF_USESHOWWINDOW; //| //STARTF_USESTDHANDLES
         si.wShowWindow = SW_HIDE;
      }
      iFlags = 0;
      
      if( overlay ) {
         si.hStdInput = GetStdHandle( STD_INPUT_HANDLE );
         si.hStdError = GetStdHandle( STD_ERROR_HANDLE );
      }
      
      si.dwFlags |= STARTF_USESTDHANDLES;
      si.hStdOutput = hWrite;
      
      bufSize = finalCmd.length() * 4 + 1;
      char *charbuf = (char *) memAlloc( bufSize );
      finalCmd.toCString( charbuf, bufSize );
      
      if ( ! CreateProcessA( NULL,
                             charbuf,
                             NULL,
                             NULL,
                             TRUE,  // INHERIT HANDLES!
                             iFlags,
                             NULL,
                             NULL,
                             &si,
			&proc
                             ) )
      {
         memFree( charbuf );
         *returnValue = GetLastError();
         return false;
      }
      
      memFree( charbuf );
	}
   
   // read from the input handle
#define read_buffer_size 4096
   char buffer[read_buffer_size];
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
   // we have to change our streams with the ones of the process.
   GetExitCodeProcess( proc.hProcess, &iRet );
   //memFree( completeCommand );
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

bool openProcess(Process* _ph, String **argv, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg )
{
   WinProcess* ph = static_cast<WinProcess*>(_ph);
   
   SECURITY_ATTRIBUTES secatt;
   
   ph->hPipeInRd=INVALID_HANDLE_VALUE;
   ph->hPipeInWr=INVALID_HANDLE_VALUE;
   ph->hPipeOutRd=INVALID_HANDLE_VALUE;
   ph->hPipeOutWr=INVALID_HANDLE_VALUE;
   ph->hPipeErrRd=INVALID_HANDLE_VALUE;
   ph->hPipeErrWr=INVALID_HANDLE_VALUE;
   
   // prepare security attributes
   secatt.nLength = sizeof( secatt );
   secatt.lpSecurityDescriptor = NULL;
   secatt.bInheritHandle = TRUE;
   
   if ( ! sinkin )
   {
      if ( !CreatePipe( &ph->hPipeInRd, &ph->hPipeInWr, &secatt, 0 ) )
      {
         ph->lastError( GetLastError() );
         return false;
      }
   }
   
   if ( ! sinkout )
   {
      if ( ! CreatePipe( &ph->hPipeOutRd, &ph->hPipeOutWr, &secatt, 0 ) )
      {
         ph->lastError( GetLastError() );
         CloseHandle( ph->hPipeInRd );
         CloseHandle( ph->hPipeInWr );
         return false;
      }
      
      if ( mergeErr ) {
         ph->hPipeErrRd = ph->hPipeOutRd;
         ph->hPipeErrWr = ph->hPipeOutWr;
      }
   }
   
   if ( ! sinkerr && ! mergeErr )
   {
      if ( !CreatePipe( &ph->hPipeErrRd, &ph->hPipeErrWr, &secatt, 0 ) )
      {
         ph->lastError( GetLastError() );
         CloseHandle( ph->hPipeInRd );
         CloseHandle( ph->hPipeInWr );
         CloseHandle( ph->hPipeOutRd );
         CloseHandle( ph->hPipeOutWr );
         return false;
      }
   }
   
   String finalCmd;
   
   // find the command in the path
   uint32 bufSize = argv[0]->length() * 4 + 1;
   wchar_t* fileNameBuf = new wchar_t[bufSize];
   argv[0]->toWideString( fileNameBuf, bufSize );
   
   bool wideImplemented;
   wchar_t* filePart_w;
   char fullCommand[2048];
   wchar_t* fullCommand_w = reinterpret_cast<wchar_t*>( fullCommand );
   AutoBuf foo(*(argv[0]), true);
   if ( ! SearchPathW( NULL, fileNameBuf, NULL, 1024, fullCommand_w, &filePart_w ) )
   {
      if ( GetLastError() == ERROR_CALL_NOT_IMPLEMENTED )
      {
         wideImplemented = false;
         char* filePart;
         char* charbuf = reinterpret_cast<char*>( fileNameBuf );
         if( argv[0]->toCString( charbuf, bufSize ) > 0 )
         {
            if ( ! SearchPathA( NULL, charbuf, NULL, 2048, fullCommand, &filePart ) )
               finalCmd.bufferize(charbuf);
            else
               finalCmd.bufferize(fullCommand);
         }
      }
      else
      {
         wideImplemented = true;
         finalCmd.bufferize(fileNameBuf);
      }
   }
   else
      finalCmd.bufferize(fullCommand_w);

   AutoBuf bar(finalCmd, true);
   delete[] fileNameBuf;
   AutoBuf bar2(finalCmd, true);
   
   // build the complete string
   for(size_t i = 1; argv[ i ] != 0; i++)
   {
      finalCmd.append( ' ' );
      finalCmd.append( *argv[i] );
   }
   AutoBuf foo2(finalCmd, true);
   
   if( wideImplemented )
   {
      PROCESS_INFORMATION proc;
      STARTUPINFOW siw;
      DWORD iFlags = 0;
      memset( &siw, 0, sizeof( siw ) );
      siw.cb = sizeof( siw );
      
      if ( ! bg )
      {
         // using show_hide AND using invalid handlers for unused streams
         siw.dwFlags = STARTF_USESTDHANDLES;
         
         siw.hStdInput = ph->hPipeInRd;
         siw.hStdOutput = ph->hPipeOutWr;
         siw.hStdError = ph->hPipeErrWr;
      }
      else
      {
         siw.dwFlags |= STARTF_USESHOWWINDOW;
         siw.wShowWindow = SW_HIDE;
         iFlags |= DETACHED_PROCESS;
      }
      
      AutoBuf cmdbuf(finalCmd, true);
      if ( ! CreateProcessW( NULL,
                             cmdbuf.wcbuf,
                             NULL,
                             NULL,
                             TRUE, //Inerhit handles!
                             iFlags,
                             NULL,
                             NULL,
                             &siw,
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
      
   }
   else
   {
      PROCESS_INFORMATION proc;      
      DWORD iFlags = 0;
      STARTUPINFOA si;
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
      
      AutoBuf cmdbuf(finalCmd, false);
      if ( ! CreateProcessA( NULL,
                             cmdbuf.cbuf,
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
