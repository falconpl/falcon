/*
   FALCON - The Falcon Programming Language.
   FILE: selector.cpp

   VM Scheduler managing waits and sleeps of contexts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Feb 2013 20:56:12 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/selector.cpp"

#include <falcon/selector.h>
#include <falcon/mt.h>
#include <falcon/stream.h>
#include <falcon/multiplex.h>

#include <map>
#include <deque>
#include <algorithm>

namespace Falcon {

class Selector::Private
{
public:
   typedef std::map<Selectable*, Selector::Data> StreamMap;
   typedef std::deque<Selector::Data*> DataList;
   typedef std::deque<Selectable*> StreamList;
   typedef std::map<const Multiplex::Factory*, Multiplex*> MultiplexMap;

   Mutex m_mtx;
   StreamMap m_selectables;
   DataList m_toBeWaited;

   StreamList m_readyRead;
   StreamList m_readyWrite;
   StreamList m_readyErr;

   int32 m_readyCount;
   int32 m_version;

   Mutex m_mtxMultiplex;
   MultiplexMap m_multiplex;
   int m_mpxVersion;

   Private(Selector* sel):
      m_readyCount(0),
      m_version(0),
      m_mpxVersion(0),
      m_selector(sel)
   {}

   ~Private()
   {
      // Data entities are in the stream map, better to check it before.
      DataList::iterator diter =  m_toBeWaited.begin();
      while( diter != m_toBeWaited.end() )
      {
         Selector::Data* data = *diter;
         data->m_resource->decref();
         ++diter;
      }

      StreamMap::iterator miter =  m_selectables.begin();
      while( miter != m_selectables.end() )
      {
         miter->first->decref();
         ++miter;
      }

      StreamList::iterator liter = m_readyRead.begin();
      while( liter != m_readyRead.end() )
      {
         Selectable* stream = *liter;
         stream->decref();
         liter++;
      }

      liter = m_readyWrite.begin();
      while( liter != m_readyWrite.end() )
      {
         Selectable* stream = *liter;
         stream->decref();
         liter++;
      }

      liter = m_readyErr.begin();
      while( liter != m_readyErr.end() )
      {
         Selectable* stream = *liter;
         stream->decref();
         liter++;
      }

      MultiplexMap::iterator mmiter = m_multiplex.begin();
      while( mmiter != m_multiplex.end() )
      {
         Multiplex* mplx = mmiter->second;
         delete mplx;

         ++mmiter;
      }
   }


   bool pushReady( Selectable* stream, StreamList& list )
   {
      stream->incref();

      bool first;
      m_mtx.lock();

      first = ( m_readyCount  == 0 );
      list.push_back(stream);
      m_readyCount++;

      m_mtx.unlock();

      return first;
   }

   Selectable* getNextReady( StreamList& list )
   {
      Selectable* stream = 0;
      bool bDesignal = false;

      m_mtx.lock();
      if( ! list.empty() )
      {
         stream = list.front();
         list.pop_front();
         m_readyCount--;
         if( m_readyCount ==  0)
         {
            bDesignal = true;
         }

         // is this stream still wanted?
         StreamMap::iterator iter = m_selectables.find( stream );
         if( iter != m_selectables.end() && iter->second.m_mode != 0 )
         {
            stream->incref();
            iter->second.m_bPending = true;
            m_toBeWaited.push_back( &iter->second );
         }
      }
      m_mtx.unlock();

      if( stream != 0 )
      {
         stream->decref();
      }

      if( bDesignal )
      {
         m_selector->Shared::consumeSignal(0, 1);
      }
      return stream;
   }

private:
   Selector* m_selector;
};


Selector::Selector(ContextManager* mgr, const Class* handler, bool acquireable ):
         Shared( mgr, handler, acquireable )
{
   _p = new Private(this);
}


Selector::~Selector()
{
   delete _p;
}


void Selector::signal( int )
{
   Shared::lockSignals();
   if( Shared::lockedSignalCount() == 0 )
   {
      Shared::lockedSignal(1);
   }
   Shared::unlockSignals();
}


int32 Selector::consumeSignal( VMContext*, int32 )
{
   // as consume signal is usually invoked upon waits, it's a good time to send the
   // streams to the waiters.
   dequePending();

   _p->m_mtx.lock();
   int32 value = _p->m_readyCount;
   _p->m_mtx.unlock();
   return value;
}


int Selector::lockedConsumeSignal( VMContext*, int )
{
   _p->m_mtx.lock();
   int32 value = _p->m_readyCount;
   _p->m_mtx.unlock();
   return value;
}


void Selector::gcMark( uint32 mark )
{
   if( m_mark != mark )
   {
      _p->m_mtx.lock();
      int32 oldVersion = _p->m_version;

      m_mark = mark;
      Private::StreamMap::iterator item = _p->m_selectables.begin();
      while( item != _p->m_selectables.end() )
      {
         Selectable* stream = item->first;
         _p->m_mtx.unlock();

         stream->gcMark(mark);

         _p->m_mtx.lock();
         if ( _p->m_version != oldVersion )
         {
            item = _p->m_selectables.begin();
            oldVersion = _p->m_version;
         }
         else {
            ++item;
         }
      }
      _p->m_mtx.unlock();
   }

}


void Selector::add( Selectable* stream, int mode, bool additive )
{
   _p->m_mtx.lock();
   Data& dt = _p->m_selectables[stream];
   // new stream around? -- data get initialized to 0
   if( dt.m_resource == 0 )
   {
      // new insertion
      _p->m_version++;
      dt.m_resource = stream;
      // one incref is for our map...
      stream->incref();
   }

   // new mode (for new and old streams)
   if( mode != 0 && dt.m_mode == 0 )
   {
      dt.m_bPending = true;
      _p->m_toBeWaited.push_back(&dt);
      // One incref is for the wait queue...
      stream->incref();
   }

   if( additive ) {
      dt.m_mode |= mode;
   }
   else {
      dt.m_mode = mode;
   }
   _p->m_mtx.unlock();
}


Selectable* Selector::getNextReadyRead()
{
   return _p->getNextReady( _p->m_readyRead );
}


Selectable* Selector::getNextReadyWrite()
{
   return _p->getNextReady( _p->m_readyWrite );
}


Selectable* Selector::getNextReadyErr()
{
   return _p->getNextReady( _p->m_readyErr );
}


void Selector::pushReadyRead( Selectable* stream )
{
   // only for the first...
   if( _p->pushReady( stream, _p->m_readyRead ) )
   {
      Shared::signal();
   }
}


void Selector::pushReadyWrite( Selectable* stream )
{
   // only for the first...
   if( _p->pushReady( stream, _p->m_readyWrite ) )
   {
      Shared::signal();
   }
}


void Selector::pushReadyErr( Selectable* stream )
{
   // only for the first...
   if( _p->pushReady( stream, _p->m_readyErr ) )
   {
      Shared::signal();
   }
}


void Selector::addRead( Selectable* stream )
{
   add(stream, mode_read);
}


void Selector::addWrite( Selectable* stream )
{
   add(stream, mode_write);
}


void Selector::addErr( Selectable* stream )
{
   add(stream, mode_err);
}


bool Selector::remove( Selectable* resource )
{
   Multiplex* mplex = 0;
   bool bFound = false;

   _p->m_mtx.lock();
   Private::StreamMap::iterator iter = _p->m_selectables.find(resource);
   if( iter != _p->m_selectables.end() )
   {
      Data& data = iter->second;
      bFound = true;
      mplex = data.m_mplex;

      if( data.m_bPending )
      {
         // Is in the waiting list.
         Private::DataList::iterator pos = std::find( _p->m_toBeWaited.begin(), _p->m_toBeWaited.end(), &data );
         if( pos != _p->m_toBeWaited.end() )
         {
            Selectable* pending = data.m_resource;
            pending->decref();
            _p->m_toBeWaited.erase( pos );
            data.m_bPending = false;
         }
      }

      _p->m_selectables.erase( iter );
      _p->m_version++;
   }
   _p->m_mtx.unlock();

   // we can safely ask for removal outside of the lock,
   // as the status of the stream in this object is regulated by the StreamMap.
   if ( bFound )
   {
      if( mplex != 0 ) {
         mplex->remove( resource );
      }
      resource->decref();
      return true;
   }

   return false;
}

void Selector::dequePending()
{
   _p->m_mtx.lock();
   while( ! _p->m_toBeWaited.empty() )
   {
      Data& data = *_p->m_toBeWaited.front();
      _p->m_toBeWaited.pop_front();
      data.m_bPending = false;
      Selectable* toBeSent = data.m_resource;
      int mode = data.m_mode;
      _p->m_mtx.unlock();

      // we can safely release the lock while searching or a multiplex.
      const Multiplex::Factory* gen = toBeSent->factory();
      Multiplex* plex;
      _p->m_mtxMultiplex.lock();
      Private::MultiplexMap::iterator pos = _p->m_multiplex.find( gen );
      if( pos == _p->m_multiplex.end() )
      {
         plex = gen->create(this);
         _p->m_multiplex[gen] = plex;
         _p->m_mpxVersion++;
      }
      else {
         plex = pos->second;
      }
      data.m_mplex = plex;
      _p->m_mtxMultiplex.unlock();

      plex->add(toBeSent, mode );
      toBeSent->decref();

      _p->m_mtx.lock();
   }
   _p->m_mtx.unlock();
}


void Selector::removeFromMultiplex( Selectable* resource )
{
   const Multiplex::Factory* gen = resource->factory();
   Multiplex* mpx = 0;

   _p->m_mtxMultiplex.lock();
   Private::MultiplexMap::iterator mpi = _p->m_multiplex.find( gen );
   if( mpi != _p->m_multiplex.end() )
   {
      mpx = mpi->second;
   }
   _p->m_mtxMultiplex.unlock();


   if( mpx != 0 )
   {
      mpx->remove(resource);

      // if this is currently the last resource, remove the multiplex.
      // this is important because, after our return, an external module where
      // the resource is allocated may be unloaded; and that module might
      // contain the multiplex code as well.
      _p->m_mtxMultiplex.lock();
      if( mpx->size() == 0 )
      {
         _p->m_multiplex.erase( mpi );
         _p->m_mtxMultiplex.unlock();
         delete mpx;
      }
      else {
         _p->m_mtxMultiplex.unlock();
      }
   }
}

}

/* end of selector.cpp */
