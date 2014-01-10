/*
   FALCON - The Falcon Programming Language.
   FILE: session.cpp

   Automatism to implement persistent data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/modules/native/feathers/shmem/session.cpp"

#include "session.h"
#include <falcon/mt.h>
#include <falcon/symbol.h>
#include <falcon/item.h>
#include <falcon/module.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/stderrors.h>
#include <falcon/stdhandlers.h>
#include <falcon/storer.h>
#include <falcon/restorer.h>
#include <falcon/itemid.h>
#include <falcon/stringstream.h>
#include <falcon/pstep.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <falcon/sys.h>

#include <map>

#include "errors.h"
#include "sharedmem.h"

namespace Falcon {

class Session::Private
{
public:
   typedef std::map<Symbol*, Item> SymbolSet;

   Mutex m_mtxSym;
   SymbolSet m_symbols;
   uint32 m_version;
   bool bInUse;

   Private( Session* session ):
      m_version(0),
      bInUse(false),
      m_stepRestore(session),
      m_stepStore(session)
   {}

   ~Private() {}

   inline void inUseCheckIn( int line )
   {
      m_mtxSym.lock();
      if( bInUse )
      {
         m_mtxSym.unlock();
         throw new SessionError(
                  ErrorParam( FALCON_ERROR_SHMEM_SESSION_CONCURRENT, line, SRC )
                   );
      }

      bInUse = true;
   }

   inline void inUseCheckOut()
   {
      bInUse = false;
      m_mtxSym.unlock();
   }


   class PStepRestore: public PStep
   {
   public:
      PStepRestore( Session* s ): m_session(s) { apply = apply_; }
      virtual ~PStepRestore() {}
      virtual void describeTo( String& s ) { s = "Session::Private::PStepRestore"; }

      static void apply_(const PStep* ps, VMContext* ctx );

   private:
      Session* m_session;
   };

   PStepRestore m_stepRestore;


   class PStepStore: public PStep
      {
      public:
         PStepStore( Session* s ): m_session(s) { apply = apply_; }
         virtual ~PStepStore() {}
         virtual void describeTo( String& s ) { s = "Session::Private::PStepStore"; }

         static void apply_(const PStep* ps, VMContext* ctx );

      private:
         Session* m_session;
      };

      PStepStore m_stepStore;
};


void Session::Private::PStepRestore::apply_(const PStep* ps, VMContext* ctx )
{
   TRACE("Session::Private::PStepRestore::apply_ with depth %d", (int) ctx->dataSize());

   const PStepRestore* self = static_cast<const PStepRestore*>(ps);
   fassert(ctx->topData().asClass()->name() == "Restorer");
   Restorer* res = static_cast<Restorer*>(ctx->topData().asInst());

   bool doApply = ctx->currentCode().m_seqId != 0;
   ctx->popCode();

   self->m_session->restore( res );

   // if everything is allright...
   ctx->returnFrame(Item().setBoolean(true));

   //ctx->popData(); // remove the restorer.

   if (doApply)
   {
      self->m_session->apply( ctx );
   }
}


void Session::Private::PStepStore::apply_(const PStep* ps, VMContext* ctx )
{
   TRACE("Session::Private::PStepRestore::apply_ with depth %d", (int) ctx->dataSize());

   const PStepStore* self = static_cast<const PStepStore*>(ps);
   fassert(ctx->topData().asClass()->name() == "Storer");
   Storer* sto = static_cast<Storer*>(ctx->topData().asInst());

   ctx->popCode();
   self->m_session->commitStore(ctx,sto);

   // if everything is allright...
   ctx->popData(); // remove the restorer.
}


//=====================================================================
// Main session object
//=====================================================================

Session::Session()
{
   init();
}


Session::Session( t_openmode mode, const String& name, int64 to ):
   m_id(name)
{
   init();
   m_timeout = to;
   m_open_mode = mode;
}

void Session::init()
{
   _p = new Private(this);
   m_timeout = 0;
   m_tsCreation = 0;
   m_tsExpire = 0;
   m_bExpired = false;
   m_stream = 0;
   m_shmem = 0;
   m_open_mode = e_om_shmem;
}

Session::~Session()
{
   if ( m_shmem != 0 )
   {
      m_shmem->close(false);
      delete m_shmem;
   }

   if( m_stream != 0 )
   {
      m_stream->decref();
   }

   delete _p;
}


void Session::getID( String& target ) const
{
   _p->m_mtxSym.lock();
   target = m_id;
   _p->m_mtxSym.unlock();
}


void Session::setID( const String& target )
{
   _p->m_mtxSym.lock();
   m_id = target;
   _p->m_mtxSym.unlock();
}


void Session::begin()
{
   int64 now = Sys::_epoch();

   _p->m_mtxSym.lock();
   if( ! m_bExpired )
   {
      m_tsCreation = now;
      if( m_timeout != 0 )
      {
         m_tsExpire = now + m_tsCreation;
      }
      else {
         m_tsExpire = 0;
      }
   }
   _p->m_mtxSym.unlock();
}


void Session::open()
{
   switch( m_open_mode )
   {
   case e_om_file:
   {
      m_stream = Engine::instance()->vfs().open(m_id, VFSProvider::OParams().rdwr() );
   }
   break;

   case e_om_shmem: case e_om_shmem_bu:
   {
      m_shmem = new SharedMem;
      m_shmem->open( m_id, m_open_mode == e_om_shmem_bu );
   }
   break;
   }

   tick();
}


void Session::create()
{
   switch( m_open_mode )
   {
   case e_om_file:
      m_stream = Engine::instance()->vfs().create(m_id, VFSProvider::CParams().rdwr().truncate() );
      break;

   case e_om_shmem: case e_om_shmem_bu:
      m_shmem = new SharedMem;
      m_shmem->open( m_id, m_open_mode == e_om_shmem_bu );
      break;
   }

   begin();
}


void Session::save( VMContext* ctx )
{
   static Class* clsStorer = Engine::instance()->stdHandlers()->storerClass();

   tick();

   _p->inUseCheckIn(__LINE__);
   // in-use throwing is a strong enough guarantee for us...
   _p->m_mtxSym.unlock();

   if ( m_shmem != 0 )
   {
      m_stream = new StringStream;
   }
   else {
      create();
   }


   const char* marker = "FALS";
   m_stream->write(marker,4);
   DataWriter writer(m_stream, DataWriter::e_LE);

   // ... but here; need stronger guarantees
   _p->m_mtxSym.lock();
   try
   {
      writer.write(m_id);
      writer.write(m_tsCreation);
      writer.write(m_tsExpire);
      writer.write(m_timeout);
      _p->m_mtxSym.unlock();
   }
   catch(...)
   {
      _p->m_mtxSym.unlock();
      throw;
   }

   // prepare the restorer for garbage collection.
   Storer* storer = new Storer;
   ctx->pushData( FALCON_GC_STORE(clsStorer, storer) );
   ctx->pushCode(&_p->m_stepStore);

   // ready to go.
   store(ctx, storer);

   // we don't care if the session is being used after this point.
   _p->m_mtxSym.lock();
   _p->inUseCheckOut();

   // let the context finish the recursive storage...
}


void Session::load( VMContext* ctx, bool bApply )
{
   static Class* clsRestorer = Engine::instance()->stdHandlers()->restorerClass();

   _p->inUseCheckIn(__LINE__);
   // in-use throwing is a strong enough guarantee for us...
   if( m_shmem == 0 && m_stream == 0 )
   {
      _p->m_mtxSym.unlock();
      open();
   }
   else {
      _p->m_mtxSym.unlock();
   }

   if ( m_stream == 0 )
   {
      fassert(m_shmem != 0);
      int64 len=0;
      void* data = m_shmem->grabAll(len);

      // still unused session file?
      if( len == 0 )
      {
         _p->m_mtxSym.lock();
         _p->inUseCheckOut();
         throw FALCON_SIGN_ERROR(SessionError, FALCON_ERROR_SHMEM_SESSION_NOTOPEN );
      }

      m_stream = new StringStream((byte*) data, len);
   }
   else {
      m_stream->seekBegin(0);
   }

   char marker[5];
   marker[4] = 0;
   if( m_stream->eof() || m_stream->read(marker,4) == 0 )
   {
      // still unused file.
      _p->m_mtxSym.lock();
      _p->inUseCheckOut();
      throw FALCON_SIGN_ERROR(SessionError, FALCON_ERROR_SHMEM_SESSION_NOTOPEN );
   }

   if ( String("FALS") != marker )
   {
      _p->m_mtxSym.lock();
      _p->inUseCheckOut();
      throw FALCON_SIGN_ERROR(SessionError, FALCON_ERROR_SHMEM_SESSION_INVALID );
   }

   DataReader reader(m_stream, DataReader::e_LE);

   String id;
   reader.read(id);
   if( id != m_id )
   {
      _p->m_mtxSym.lock();
      _p->inUseCheckOut();
      throw FALCON_SIGN_ERROR(SessionError, FALCON_ERROR_SHMEM_SESSION_INVALID );
   }

   // ... but here; need stronger guarantees
   _p->m_mtxSym.lock();
   try
   {
      reader.read(m_tsCreation);
      reader.read(m_tsExpire);
      reader.read(m_timeout);
      _p->m_mtxSym.unlock();

      reader.resetStream();
   }
   catch(...)
   {
      _p->inUseCheckOut();
      throw;
   }

   if( isExpired() )
   {
      _p->m_mtxSym.lock();
      _p->inUseCheckOut();
      throw FALCON_SIGN_ERROR(SessionError, FALCON_ERROR_SHMEM_SESSION_EXPIRED );
   }

   // prepare the restorer for garbage collection.
   Restorer* rest = new Restorer;
   ctx->pushData( FALCON_GC_STORE(clsRestorer, rest) );
   ctx->pushCode(&_p->m_stepRestore);
   TRACE1("Session::load pushed stepRestore with depth %d", (int) ctx->dataSize());
   if( bApply ) {
      ctx->currentCode().m_seqId = 1;
   }

   // ready to go.
   rest->restore(ctx, m_stream, ctx->process()->modSpace());

   // we don't care if the session is being used after this point.
   _p->m_mtxSym.lock();
   _p->inUseCheckOut();
}


void Session::close()
{
   if (m_shmem != 0)
   {
      m_shmem->close(true);
      m_shmem = 0;
   }

   if( m_stream != 0 )
   {
      m_stream->decref();
      m_stream = 0;

      if( m_open_mode == e_om_file )
      {
         // this might throw...
         Engine::instance()->vfs().erase(m_id);
      }
   }
}



void Session::addSymbol(Symbol* sym, const Item& value)
{
   _p->inUseCheckIn(__LINE__ );

   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   _p->m_version++;
   if( iter == _p->m_symbols.end() )
   {
      sym->incref();
      _p->m_symbols.insert(std::make_pair(sym, value));
   }
   else {
      iter->second = value;
   }

   _p->inUseCheckOut();
}


bool Session::removeSymbol(Symbol* sym)
{
   _p->inUseCheckIn(__LINE__ );

   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   if( iter != _p->m_symbols.end() )
   {
      _p->m_version++;
      _p->m_symbols.erase(iter);
      _p->inUseCheckOut();

      sym->decref();
      return true;
   }
   else {
      _p->inUseCheckOut();
   }

   return false;
}


void Session::record( VMContext* ctx )
{
   _p->inUseCheckIn(__LINE__);
   _p->m_version++;
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.end() )
   {
      Symbol* sym = iter->first;
      try {
         Item* item = ctx->resolveSymbol(sym, false);
         if( item != 0 )
         {
            iter->second.copyFromRemote(*item);
         }
      }
      catch( CodeError* e )
      {
         e->decref();
      }
      catch( Error* e )
      {
         e->decref();
      }

      ++iter;
   }
   _p->inUseCheckOut();
}


void Session::apply( VMContext* ctx ) const
{
   _p->inUseCheckIn(__LINE__);

   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.end() )
   {
      Symbol* sym = iter->first;
      Item* item = ctx->resolveSymbol(sym, true);
      fassert( *item != 0 );
      item->copyFromLocal(iter->second);
      ++iter;
   }

   _p->inUseCheckOut();
}


void Session::store(VMContext* ctx, Storer* storer) const
{
   Class* cls = 0;
   void* data = 0;

   // We don't lock here, as we have a concurrent prevention mechanism in the
   // invoking method.
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.end() )
   {
      Symbol* sym = iter->first;
      storer->store(ctx, sym->handler(), sym, false);
      iter->second.forceClassInst(cls,data);
      // the data is in GC, as we know we have locked it.
      storer->store(ctx, cls, data, true);
      ++iter;
   }

   // write a nil as end marker
   Item item;
   item.forceClassInst(cls, data);
   storer->store(ctx, cls, data, false);
}


void Session::commitStore(VMContext* ctx, Storer* sto)
{
   fassert( m_stream != 0 );
   _p->inUseCheckIn(__LINE__);
   _p->m_mtxSym.unlock();

   sto->commit(ctx, m_stream);

   if( m_shmem != 0 )
   {
      StringStream* ss = static_cast<StringStream*>(m_stream);
      String temp;
      ss->closeToString(temp);
      m_shmem->write(temp.getRawStorage(), temp.size(), 0, false, true);
   }

   m_stream->close();

   _p->m_mtxSym.lock();
   m_stream = 0;
   _p->inUseCheckOut();
}


void Session::restore(Restorer* restorer)
{
   static Class* symClass = Engine::instance()->stdHandlers()->symbolClass();

   Class* handler = 0;
   void* data = 0;
   bool first = false;
   while( restorer->next(handler,data,first) )
   {
      if( handler->typeID() == FLC_ITEM_NIL )
      {
         // we're done
         return;
      }

      // the first should be a symbol
      if( handler != symClass )
      {
         throw FALCON_SIGN_XERROR(IOError, e_deser, .extra("Missing leading symbol in session restore") );
      }
      Symbol* sym = static_cast<Symbol*>(data);

      // and the second is our item.
      if( ! restorer->next(handler, data, first) )
      {
         throw FALCON_SIGN_XERROR(IOError, e_deser, .extra("Missing data in session restore") );
      }

      // flat data?
      if( handler->typeID() < FLC_ITEM_COUNT )
      {
         Item value = *static_cast<Item*>(data);
         addSymbol( sym, value );
      }
      else {
         Item value(handler, data);
         addSymbol( sym, value );
      }

   }
}


bool Session::get(Symbol* sym, Item& item) const
{
   bool res = false;
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   if( iter != _p->m_symbols.end() )
   {
      item = iter->second;
      res = true;
   }
   _p->m_mtxSym.unlock();

   return res;
}


bool Session::get(const String& symName, Item& item) const
{
   Symbol* sym = Engine::getSymbol(symName);
   bool res = get(sym, item);
   sym->decref();
   return res;
}


int64 Session::createdAt() const
{
   _p->m_mtxSym.lock();
   int64 c = m_tsCreation;
   _p->m_mtxSym.unlock();
   return c;
}


int64 Session::expiresAt() const
{
   _p->m_mtxSym.lock();
   int64 c = m_tsExpire;
   _p->m_mtxSym.unlock();
   return c;
}


int64 Session::timeout() const
{
   _p->m_mtxSym.lock();
   int64 c = m_timeout;
   _p->m_mtxSym.unlock();
   return c;
}


void Session::timeout( int64 to )
{
   int64 timeNow = 0;
   if( to > 0 )
   {
      timeNow = Sys::_epoch();

      _p->m_mtxSym.lock();
      m_timeout = to;
      m_tsExpire = timeNow + to;
      _p->m_mtxSym.unlock();
   }
   else {
      _p->m_mtxSym.lock();
      m_timeout = 0;
      m_tsExpire = 0;
      _p->m_mtxSym.unlock();
   }
}


void Session::tick()
{
   int64 timeNow = Sys::_epoch();

   _p->m_mtxSym.lock();
   if( m_timeout > 0 )
   {
      m_tsExpire = timeNow + m_timeout;
   }
   _p->m_mtxSym.unlock();
}


bool Session::isExpired() const
{
   // optimistic check on a locked variable.
   // expired can change only from false to true.
   if( m_bExpired )
   {
      return true;
   }

   int64 timeNow = Sys::_epoch();
   bool bExpired = false;

   _p->m_mtxSym.lock();
   if( m_tsExpire > 0 && timeNow > m_tsExpire )
   {
      bExpired = true;
      m_bExpired = true;
   }
   _p->m_mtxSym.unlock();

   return bExpired;
}


void Session::gcMark( uint32 mark )
{
   if( m_mark == mark )
   {
      return;
   }

   m_mark = mark;

   _p->m_mtxSym.lock();
   uint32 version = _p->m_version;
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.end() )
   {
      Item item = iter->second;
      _p->m_mtxSym.unlock();

      item.gcMark(mark);

      // mark again from the start if the map changed.
      _p->m_mtxSym.lock();
      if( version == _p->m_version )
      {
         ++iter;
      }
      else {
         iter = _p->m_symbols.begin();
         version = _p->m_version;
      }
   }
   _p->m_mtxSym.unlock();
}


int64 Session::occupiedMemory() const
{
   int64 count;
   _p->m_mtxSym.lock();
   count = _p->m_symbols.size() * 16;
   _p->m_mtxSym.unlock();

   return sizeof(Session) + 16 + count;
}


void Session::enumerate( Enumerator& r ) const
{
   _p->inUseCheckIn(__LINE__);
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.end() )
   {
      Symbol* sym = iter->first;
      Item& value = iter->second;
      r(sym,value);
      ++iter;
   }

   _p->inUseCheckOut();
}

bool Session::checkLoad() const
{
   bool check = false;
   _p->inUseCheckIn(__LINE__);
   if( m_shmem != 0 )
   {
      check = m_shmem->size() > 0;
   }
   else if (m_stream != 0) {
      Stream* stream = m_stream;
      stream->incref();
      _p->m_mtxSym.unlock();

      check = stream->seekEnd(0) != 0;
      stream->decref();
      _p->m_mtxSym.lock();
   }
   _p->inUseCheckOut();

   return check;
}

}

/* end of session.cpp */
