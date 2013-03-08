/*
   FALCON - The Falcon Programming Language.
   FILE: filedatampx_win.cpp

   Multiplexer for blocking system-level streams -- Windows
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 11:20:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/filedatampx_win.cpp"
#define _WIN32_WINNT 0x0500

#include <falcon/filedatampx.h>
#include <falcon/selector.h>
#include <falcon/fstream.h>

#include <falcon/errors/genericerror.h>
#include <falcon/errors/ioerror.h>

#define POLL_THREAD_STACK_SIZE (32*1024)

#include <map>
#include <deque>

#include <windows.h>


namespace Falcon {
namespace Sys {

class FileDataMPX::Private
{
public:

   struct Msg
   {
      int32 type; // 0 = quit, 1 = add, 2=remove
      int32 mode;
      Stream* stream;

      Msg():
         type(0),
         mode(0),
         stream(0)
      {};

      Msg( int32 t, int32 m=0, Stream* s=0 ):
         type(t),
         mode(m),
         stream(s)
         {}
   };

   typedef std::deque<Msg> MessageQueue;
   typedef std::map<Stream*, int> StreamMap;

   MessageQueue m_messages;
   StreamMap m_streams;
   HANDLE m_hEventWakeup;
   HANDLE m_hThread;
   CRITICAL_SECTION m_csLock;
   CRITICAL_SECTION m_busyLock;

   FileDataMPX* m_master;

   static DWORD WINAPI ThreadProc( 
        LPVOID lpParameter
   );

   Private( FileDataMPX* master )
   {      
      m_master = master;
      InitializeCriticalSectionAndSpinCount(&m_csLock, 250);
      InitializeCriticalSectionAndSpinCount(&m_busyLock, 250);
      m_hEventWakeup = CreateEvent(NULL, FALSE, FALSE, NULL);      
   }

   ~Private()
   {
      CloseHandle(m_hThread);
      CloseHandle(m_hEventWakeup);
      DeleteCriticalSection(&m_csLock);
      DeleteCriticalSection(&m_busyLock);
   }

   void start()
   {
      DWORD ThreadId = 0;
      m_hThread = CreateThread(
         NULL,
         POLL_THREAD_STACK_SIZE,
         &ThreadProc,
         this,
         0,
         &ThreadId
         );
   }

   void sendMessage( const Msg& msg )
   {
      EnterCriticalSection( &m_csLock );
      m_messages.push_back( msg );
      LeaveCriticalSection( &m_csLock );      
      SetEvent( m_hEventWakeup );
   }

   void quit()
   {
      sendMessage(Msg(0));      
      // wait for our thread to finish.
      WaitForSingleObject( m_hThread, INFINITE );      
   }

   void addToMultiplex( Stream* strem, int mode );
   void removeFromMultiplex( Stream* stream );
   void removeStreams();

   static VOID CALLBACK onReadComplete(
      DWORD dwErrorCode,
      DWORD dwNumberOfBytesTransfered,
      LPOVERLAPPED lpOverlapped
   );

   static VOID CALLBACK onWriteComplete(
      DWORD dwErrorCode,
      DWORD dwNumberOfBytesTransfered,
      LPOVERLAPPED lpOverlapped
   );

   static VOID CALLBACK onWaitReadComplete(
      PVOID lpParameter,
      BOOLEAN TimerOrWaitFired
   );

   static VOID CALLBACK onWaitWriteComplete(
      PVOID lpParameter,
      BOOLEAN TimerOrWaitFired
   );
};


FileDataMPX::FileDataMPX( const StreamTraits* generator, Selector* master ):
         Multiplex( generator, master )
{
   _p = new Private( this );   
   _p->start();
}


FileDataMPX::~FileDataMPX()
{
   _p->quit();
   delete _p;
}


void FileDataMPX::addStream( Stream* stream, int mode )
{
   stream->incref();
   // Mode 1 to add
   _p->sendMessage( Private::Msg( 1, mode, stream ) );
}

void FileDataMPX::removeStream( Stream* stream )
{
   stream->incref();
   // Mode 2 to remove
   _p->sendMessage( Private::Msg( 2, 0, stream ) );
}


DWORD FileDataMPX::Private::ThreadProc( LPVOID data )
{
   FileDataMPX::Private* self = static_cast<FileDataMPX::Private*>(data);

   while( true )
   {
      Msg message;

      // wait for new messages
      DWORD waitResult = WaitForSingleObjectEx( self->m_hEventWakeup, INFINITE, TRUE );
      if( waitResult == WAIT_FAILED )
      {
         throw new GenericError( ErrorParam( e_io_invalid, __LINE__, SRC )
            .extra("Wait failed")
            .sysError(GetLastError())
            );
      }

      EnterCriticalSection( &self->m_csLock );
      if( self->m_messages.empty() )
      {
         // spurious wakeup?
         LeaveCriticalSection( &self->m_csLock );
         continue; 
      }

      message = self->m_messages.front();
      self->m_messages.pop_front();
      LeaveCriticalSection( &self->m_csLock );

      // process the incoming messages.
      if( message.type == 0)
      {
         // we're done
         break;
      }
      else if( message.type == 1 )
      {
         self->addToMultiplex( message.stream, message.mode );
         // keep the extra reference.
      }
      else if( message.type == 2 )
      {
         self->removeFromMultiplex( message.stream );
         // we're not keeping the stream
         message.stream->decref();
      }
   }

   self->removeStreams();
   return 0;
}


void FileDataMPX::Private::addToMultiplex( Stream* stream, int mode )
{
   // just extra?
   if( mode == Selector::mode_err )
   {
      // will never be signaled.
      stream->decref();
      return;
   }

   if( ! m_streams.insert( std::make_pair(stream, mode) ).second )
   {
      // don't need to keep the reference we received in the message
      stream->decref();
   }
   
   FStream* fs = static_cast<FStream*>(stream);
   FileDataEx* fdx = static_cast<FileDataEx*>(fs->fileData());

   if( (mode & Selector::mode_read) != 0 )
   {

      fdx->ovl.owner = stream;
      fdx->ovl.extra = this;
      if( fdx->bConsole )
      {
         // first time?         
         if( fdx->hEmulWrite == INVALID_HANDLE_VALUE )
         {
            // initialize the console work
            fdx->hRealConsole = fdx->hFile;
            CreatePipe(&fdx->hFile, &fdx->hEmulWrite, NULL, 0);
         }

         // anonymous pipe or similar
         HANDLE newWaitObject = 0;
         //DWORD  test = WaitForSingleObject(fdx->hFile, INFINITE);
         BOOL res = RegisterWaitForSingleObject(  
            &newWaitObject,
            fdx->hRealConsole,
            &onWaitReadComplete,
            fdx,
            INFINITE,
            WT_EXECUTEINWAITTHREAD | WT_EXECUTEONLYONCE
         );

         if( ! res ) 
         {
            // we got an error.
            // this will actually assert...
            throw new GenericError( ErrorParam( e_io_error, __LINE__, SRC)
               .extra("Error in RegisterWaitForSingleObject")
               .sysError( GetLastError() ) );
         }
      }
      else 
      {
         BOOL res = ReadFileEx( fdx->hFile, NULL, 0, &fdx->ovl.overlapped, &onReadComplete );
         if( ! res ) 
         {
            DWORD err = GetLastError();
            if( err != ERROR_IO_PENDING )
            {
               // declare ready for read to let the error go.
               m_master->onReadyRead( stream );
            }
         }
      }
   }

   if( (mode & Selector::mode_write) != 0 )
   {
      fdx->ovl.owner = stream;
      fdx->ovl.extra = this;
      if( fdx->bConsole )
      {
         // anonymous pipe or similar
         HANDLE newWaitObject = 0;
         //DWORD  test = WaitForSingleObject(fdx->hFile, INFINITE);
         BOOL res = RegisterWaitForSingleObject(  
            &newWaitObject,
            fdx->hFile,
            &onWaitWriteComplete,
            fdx,
            INFINITE,
            WT_EXECUTEINWAITTHREAD | WT_EXECUTEONLYONCE
         );

         if( ! res ) 
         {
            // we got an error.
            // this will actually assert...
            throw new GenericError( ErrorParam( e_io_error, __LINE__, SRC)
               .extra("Error in RegisterWaitForSingleObject")
               .sysError( GetLastError() ) );
         }
      }
      else {
         BOOL res = WriteFileEx( fdx->hFile, 0, 0, &fdx->ovl.overlapped, &onReadComplete );
         if( ! res ) 
         {
            DWORD err = GetLastError();
            if( err != ERROR_IO_PENDING )
            {
               // declare ready for read to let the error go.
               m_master->onReadyWrite( stream );
            }
         }
      }
   }
}


void FileDataMPX::Private::removeFromMultiplex( Stream* stream )
{
   if( m_streams.erase(stream) > 0 )
   {
      FStream* fs = static_cast<FStream*>(stream);
      FileDataEx* fdx = static_cast<FileDataEx*>(fs->fileData());
      EnterCriticalSection( &m_busyLock );
      if( fdx->bBusy )
      {
         fdx->bBusy = false;
         LeaveCriticalSection( &m_busyLock );
         
         //CancelIoEx(fdx->hFile, &fdx->ovl.overlapped);
         CancelIo(fdx->hFile);
      }
      else 
      {
         LeaveCriticalSection( &m_busyLock );
      }

      
      // remove the reference we had in the map
      stream->decref();
   }
}


void FileDataMPX::Private::removeStreams()
{
   StreamMap::iterator iter = m_streams.begin();
   while( iter != m_streams.end() )
   {
      FStream* fs = static_cast<FStream*>(iter->first);
      FileDataEx* fdx = static_cast<FileDataEx*>(fs->fileData());
      if( fdx->bBusy )
      {
         //CancelIoEx(fdx->hFile, &fdx->ovl.overlapped);
         CancelIo(fdx->hFile);
      }
      fs->decref();
      ++iter;
   }
}


VOID CALLBACK FileDataMPX::Private::onReadComplete(
      DWORD dwErrorCode,
      DWORD dwNumberOfBytesTransfered,
      LPOVERLAPPED lpOverlapped
)
{
   FileDataEx::OVERLAPPED_EX* ovl = reinterpret_cast<FileDataEx::OVERLAPPED_EX*>(lpOverlapped);
   FileDataMPX::Private* self = static_cast<FileDataMPX::Private*>(ovl->extra);
   self->m_master->onReadyRead(ovl->owner);
}


VOID CALLBACK FileDataMPX::Private::onWriteComplete(
      DWORD dwErrorCode,
      DWORD dwNumberOfBytesTransfered,
      LPOVERLAPPED lpOverlapped
)
{
   FileDataEx::OVERLAPPED_EX* ovl = reinterpret_cast<FileDataEx::OVERLAPPED_EX*>(lpOverlapped);
   FileDataMPX::Private* self = static_cast<FileDataMPX::Private*>(ovl->extra);
   self->m_master->onReadyWrite(ovl->owner);
}


VOID CALLBACK FileDataMPX::Private::onWaitReadComplete(
      PVOID lpParameter,
      BOOLEAN /* TimerOrWaitFired */
)
{
   FileDataEx* fdx = reinterpret_cast<FileDataEx*>(lpParameter);
   FileDataMPX::Private* self = static_cast<FileDataMPX::Private*>(fdx->ovl.extra);
   // read the event
   INPUT_RECORD Input[10];
   DWORD readElems = 0;

   BOOL res = PeekConsoleInput(fdx->hRealConsole, Input, 10, &readElems);
   if( res == FALSE )
   {
      readElems = GetLastError();
      return;
   }

   for( DWORD i = 0; i < readElems; ++i )
   {
      if( Input[i].EventType == KEY_EVENT )
      {
         if( ! Input[i].Event.KeyEvent.bKeyDown )
         {
            char chr = Input[i].Event.KeyEvent.uChar.AsciiChar;
            DWORD count;
            if( chr == 13 )
            {
               // the owner can read.
               ReadConsoleInput(fdx->hRealConsole, Input, i+1, &readElems);
               WriteFile(fdx->hEmulWrite, &chr, 1, &count, NULL );
               self->m_master->onReadyRead(fdx->ovl.owner);
               return;
            }
            else {
               WriteFile(fdx->hEmulWrite, &chr, 1, &count, NULL );
            }
         }
      } 
   }

   if( readElems > 0 )
   {
      ReadConsoleInput(fdx->hRealConsole, Input, readElems, &readElems);
   }

   // put me back on read.
   fdx->ovl.owner->incref();
   self->addToMultiplex( fdx->ovl.owner, Selector::mode_read );
}

VOID CALLBACK FileDataMPX::Private::onWaitWriteComplete(
      PVOID lpParameter,
      BOOLEAN /* TimerOrWaitFired */
)
{
   FileDataEx* fdx = reinterpret_cast<FileDataEx*>(lpParameter);
   FileDataMPX::Private* self = static_cast<FileDataMPX::Private*>(fdx->ovl.extra);
   self->m_master->onReadyWrite(fdx->ovl.owner);   
}  

}
}

/* end of filedatampx_win.cpp */
