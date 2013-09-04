/*
   FALCON - The Falcon Programming Language.
   FILE: filedatampx_posix.cpp

   Multiplexer for blocking system-level streams -- Generic POSIX
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 11:20:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/filedatampx_posix.cpp"

#include <falcon/filedatampx.h>
#include <falcon/mt.h>
#include <falcon/selector.h>
#include <falcon/fstream.h>

#include <falcon/stderrors.h>

#define POLL_THREAD_STACK_SIZE (32*1024)

#include <map>

#include <unistd.h>
#include <errno.h>
#include <poll.h>


namespace Falcon {
namespace Sys {

class FileDataMPX::Private
{
public:

   class MPXThread: public Runnable
   {
   public:
      MPXThread( FileDataMPX* master, int ctrl ):
         m_master( master ),
         m_ctrl(ctrl)
      {}

      virtual ~MPXThread() {
         int res = ::close(m_ctrl);
         if( res < 0 )
         {
            throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
                     .extra("Cannot close control pipe" ) );
         }
      }

      virtual void* run();

   private:
      typedef std::map<int, std::pair<Selectable*, int> > StreamMap;
      StreamMap m_streams;

      FileDataMPX* m_master;
      // stream where we receive control messages.
      int m_ctrl;

      bool processMessage();
   };

   struct Msg
   {
      int32 type; // 0 = quit, 1 = add, 2=remove
      int32 mode;
      Selectable* resource;
   };

   SysThread* m_thread;
   int m_ctrl;

   Private( FileDataMPX* master )
   {
      int fd[2];
      int result = pipe(fd);
      if( result != 0 )
      {
         throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("Cannot open control pipe" ) );
      }

      m_ctrl = fd[1];
      m_thread = new SysThread( new MPXThread(master, fd[0] ) );
   }

   ~Private()
   {}

   void quit()
   {
      Msg message;
      message.type = 0;
      int res = ::write( m_ctrl, &message, sizeof(message) );
      if( res < 0 )
      {
         throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("Cannot write termination message to control pipe" ) );
      }

      res = ::close(m_ctrl);
      if( res < 0 )
      {
         throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("Cannot close control pipe" )
                  .sysError((uint32)errno) );
      }

      void* dummy = 0;
      m_thread->join(dummy);
   }
};



FileDataMPX::FileDataMPX( const Multiplex::Factory* generator, Selector* master ):
         Multiplex( generator, master ),
         m_size(0)
{
   _p = new Private( this );
   _p->m_thread->start(ThreadParams().stackSize(POLL_THREAD_STACK_SIZE));
}

FileDataMPX::~FileDataMPX()
{
   _p->quit();

   delete _p;
}

void FileDataMPX::add( Selectable* resource, int mode )
{
   Private::Msg msg;
   msg.type = 1; // add
   msg.mode = mode;
   msg.resource = resource;
   resource->incref();

   int res = ::write( _p->m_ctrl, &msg, sizeof(msg) );
   if( res < 0 )
   {
      throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
               .extra("Cannot close control pipe" )
               .sysError((uint32)errno) );
   }

   m_size++;
}

void FileDataMPX::remove( Selectable* resource )
{
   Private::Msg msg;
   msg.type = 2; // remove
   msg.resource = resource;
   resource->incref();

   int res = ::write( _p->m_ctrl, &msg, sizeof(msg) );
   if( res < 0 )
   {
      throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
               .extra("Cannot close control pipe" )
               .sysError((uint32)errno) );
   }

   m_size--;
}

uint32 FileDataMPX::size() const
{
   return m_size;
}

void* FileDataMPX::Private::MPXThread::run()
{
   size_t allocated = 0;
   struct pollfd *fds = 0;
   StreamMap::iterator iter;

   while( true )
   {
      // realloc?
      if( allocated < 1+m_streams.size() )
      {
         delete[] fds;
         fds = new pollfd[1+m_streams.size()];
         allocated = 1+m_streams.size();
      }

      // prepare the poll call
      iter = m_streams.begin();
      int count = 1;
      fds[0].fd = m_ctrl;
      fds[0].events = POLLIN;
      fds[0].revents = 0;

      while( iter != m_streams.end() )
      {
         pollfd& current = fds[count++];
         current.fd = iter->first;
         current.events = 0;
         current.revents = 0;

         int mode = iter->second.second;
         if( (mode & Selector::mode_read) )
         {
            current.events |= POLLIN;
         }
         if( (mode & Selector::mode_write) )
         {
            current.events |= POLLOUT;
         }
         if( (mode & Selector::mode_err) )
         {
            current.events |= POLLPRI;
            // other interesting poll events are always listened
         }

         ++iter;
      }

      int signaled = poll( fds, count, -1 );
      if( signaled == -1 )
      {
         delete[] fds;
         iter = m_streams.begin();
         while( iter != m_streams.end() )
         {
            iter->second.first->decref();
         }

         throw new IOError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("During poll")
                  .sysError((uint32)errno)
                  );
      }

      // should we sign off?
      if ( (fds[0].revents & POLLIN) != 0 )
      {
         --signaled;
         if ( ! processMessage() )
         {
            break;
         }
      }

      count = 1;
      while( signaled > 0 )
      {
         pollfd& current = fds[count++];
         Selectable* resource = m_streams[current.fd].first;

         bool bSignaled = false;
         if( (current.revents & (POLLIN | POLLERR| POLLHUP)) != 0 )
         {
            m_master->onReadyRead(resource);
            bSignaled = true;
         }

         if( (current.revents & (POLLOUT)) != 0 )
         {
            m_master->onReadyWrite(resource);
            bSignaled = true;
         }

         if( (current.revents & (POLLPRI)) != 0 )
         {
            m_master->onReadyErr(resource);
            bSignaled = true;
         }

         if( bSignaled )
         {
            --signaled;
            resource->decref();
            m_streams.erase( current.fd );
         }
      }
   }

   delete[] fds;
   iter = m_streams.begin();
   while( iter != m_streams.end() )
   {
      iter->second.first->decref();
      ++iter;
   }

   return 0;
};


bool FileDataMPX::Private::MPXThread::processMessage()
{
   Private::Msg msg;
   uint32 count = 0;
   while( count < sizeof(msg) )
   {
      int res = ::read( m_ctrl, &msg+count, sizeof( msg )-count );

      if( res <= 0 )
      {
         throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("Error while reading from internal pipe" ) );
      }
      count += res;
   }

   FStream* fs = static_cast<FStream*>(msg.resource->instance());
   switch( msg.type )
   {
   case 0:
      return false;

   case 1:
      m_streams[fs->fileData()->fdFile] = std::make_pair( msg.resource, msg.mode );
      return true;

   case 2:
      m_streams.erase( fs->fileData()->fdFile );
      fs->decref();
      // yep, it's the same, but we have an extra ref in the message...
      msg.resource->decref();
      return true;
   }

   return false;
}

}
}

/* end of filedatampx_posix.cpp */
