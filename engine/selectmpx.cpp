/*
   FALCON - The Falcon Programming Language.
   FILE: selectmpx.cpp

   Multiplexer for blocking system-level streams -- using SELECT

   This multicplexer is used on those systems and for those 
   circumstances where a system-level file descriptor is available,
   and the select() interface is available as well.

   This includes, among other things, the MS-Windows version of the
   socket selector and OSX version of the file data selector.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 11:20:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/selectmpx.cpp"

#include <falcon/setup.h>

#ifdef FALCON_SYSTEM_WIN
#include <Winsock2.h>
#else
#include <unistd.h>
#include <errno.h>
#include <poll.h>
#endif

#include <falcon/selectmpx.h>
#include <falcon/mt.h>
#include <falcon/selector.h>
#include <falcon/fstream.h>

#include <falcon/stderrors.h>

#define POLL_THREAD_STACK_SIZE (32*1024)

#include <map>

namespace Falcon {
namespace Sys {

//================================================================
// Factory for SelectMPX
//================================================================

class SelectMPX::Private
{
public:

   class MPXThread: public Runnable
   {
   public:
      MPXThread( SelectMPX* master ):
         m_master( master )
      {}

      virtual ~MPXThread() {
      }

      virtual void* run();

   private:
      typedef std::map<int, std::pair<Selectable*, int> > StreamMap;
      StreamMap m_streams;

      SelectMPX* m_master;

      bool processMessage();
   };

   struct Msg
   {
      int32 type; // 0 = quit, 1 = add, 2=remove
      int32 mode;
      Selectable* resource;
   };

   SysThread* m_thread;

   Private( SelectMPX* master )
   {
      m_master = master;
      m_thread = new SysThread( new MPXThread( master ) );
   }

   ~Private()
   {  
   }

   void quit()
   {
      Msg message;
      message.type = 0;
      int res = m_master->writeControlFD( &message, sizeof(message) );
      if( res < 0 )
      {
         throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("Cannot write termination message to control pipe" ) );
      }

      void* dummy = 0;
      m_thread->join(dummy);
   }

   SelectMPX* m_master;
};



SelectMPX::SelectMPX( const Multiplex::Factory* generator, Selector* master ):
         Multiplex( generator, master ),
         m_size(0)
{

   _p = new Private( this );
   _p->m_thread->start(ThreadParams().stackSize(POLL_THREAD_STACK_SIZE));
}


void SelectMPX::quit()
{ 
   _p->quit();
}


SelectMPX::~SelectMPX()
{
   delete _p;
}

void SelectMPX::add( Selectable* resource, int mode )
{
   Private::Msg msg;
   msg.type = 1; // add
   msg.mode = mode;
   msg.resource = resource;
   resource->incref();
   
   int res = writeControlFD( &msg, sizeof(msg) );
   if( res < 0 )
   {
      throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
               .extra("Cannot close control pipe" )
               .sysError((uint32)errno) );
   }

   m_size++;
}

void SelectMPX::remove( Selectable* resource )
{
   Private::Msg msg;
   msg.type = 2; // remove
   msg.resource = resource;
   resource->incref();

   int res = writeControlFD( &msg, sizeof(msg) );
   if( res < 0 )
   {
      throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
               .extra("Cannot close control pipe" )
               .sysError((uint32)errno) );
   }

   m_size--;
}

uint32 SelectMPX::size() const
{
   return m_size;
}

void* SelectMPX::Private::MPXThread::run()
{
   StreamMap::iterator iter;   
   struct fd_set read_set;
   struct fd_set write_set;
   struct fd_set err_set;

   SelectMPX::FILE_DESCRIPTOR ctrl = m_master->getSelectableControlFD();
   while( true )
   {
      // prepare the zero everything
      FD_ZERO(&read_set);
      FD_ZERO(&write_set);
      FD_ZERO(&err_set);

      // add the control descriptor to the read area.
      FD_SET( ctrl, &read_set );
      SelectMPX::FILE_DESCRIPTOR maxFD = ctrl;

      iter = m_streams.begin();
      while( iter != m_streams.end() )
      {
         SelectMPX::FILE_DESCRIPTOR fd = iter->first;
         int mode = iter->second.second;
         bool added = false;

         if( (mode & Selector::mode_read) )
         {
            FD_SET( fd, &read_set );
            added = true;
         }
         if( (mode & Selector::mode_write) )
         {
            FD_SET( fd, &write_set );
            added = true;
         }
         if( (mode & Selector::mode_err) )
         {
            FD_SET( fd, &err_set );
            added = true;
         }

         if( added ) {
            if( maxFD < fd ) {
               maxFD = fd;
            }
         }

         ++iter;
      }

      int signaled = select( maxFD + 1, &read_set, &write_set, &err_set, NULL );
      if( signaled == -1 )
      {
         iter = m_streams.begin();
         while( iter != m_streams.end() )
         {
            iter->second.first->decref();
         }

         throw new IOError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("During select")
                  .sysError((uint32)errno)
                  );
      }

      // should we sign off?
      if ( FD_ISSET( ctrl, &read_set ) )
      {
         --signaled;
         if ( ! processMessage() )
         {
            break;
         }
      }

      iter = m_streams.begin();
      while( signaled > 0 && iter != m_streams.end() )
      {
         SelectMPX::FILE_DESCRIPTOR fd = iter->first;
         Selectable* resource = m_streams[fd].first;

         bool bSignaled = false;
         if( FD_ISSET( fd, &read_set ) )
         {
            m_master->onReadyRead(resource);
            bSignaled = true;
         }
         if( FD_ISSET( fd, &write_set ) )
         {
            m_master->onReadyWrite(resource);
            bSignaled = true;
         }
         if( FD_ISSET( fd, &err_set ) )
         {
            m_master->onReadyErr(resource);
            bSignaled = true;
         }

         if( bSignaled )
         {
            --signaled;
         }

         ++iter;
      }
   }

   iter = m_streams.begin();
   while( iter != m_streams.end() )
   {
      iter->second.first->decref();
      ++iter;
   }

   return 0;
};


bool SelectMPX::Private::MPXThread::processMessage()
{
   Private::Msg msg;
   uint32 count = 0;

   while( count < sizeof(msg) )
   {
      int res = m_master->readControlFD( &msg + count, sizeof( msg )-count );

      if( res <= 0 )
      {
         throw new GenericError( ErrorParam(e_io_error, __LINE__, SRC)
                  .extra("Error while reading from internal pipe" ) );
      }
      count += res;
   }

   FDSelectable* fdsel = static_cast<FDSelectable*>( msg.resource );
   int fd = 0;
   switch( msg.type )
   {
   case 0:
      return false;

   case 1:
      fd = fdsel->getFd();
      m_streams[fd] = std::make_pair( msg.resource, msg.mode );
      return true;

   case 2:
      fd = fdsel->getFd();
      m_streams.erase( fd );
      msg.resource->decref();
      return true;
   }

   return false;
}


}
}

/* end of SelectMPX.cpp */
