/*
   FALCON - The Falcon Programming Language.
   FILE: session.cpp

   Falcon script interface for Inter-process semaphore.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 15 Nov 2013 15:31:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

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

#include <falcon/sys.h>

#include <map>

namespace Falcon {

class Session::Private
{
public:
   typedef std::map<Symbol*, Item> SymbolSet;

   Mutex m_mtxSym;
   SymbolSet m_symbols;
   uint32 m_version;

   Private():
      m_version(0)
   {}

   ~Private() {}
};


Session::Session()
{
   _p = new Private;
   m_timeout = 0;
   m_tsCreation = 0;
   m_tsExpire = 0;
   m_bExpired = false;
}


Session::Session( const String& name, int64 to ):
   m_id(name)
{
   _p = new Private;
   m_timeout = 0;
   m_tsCreation = to;
   m_tsExpire = 0;
   m_bExpired = false;
}


Session::~Session()
{
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


void Session::start()
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


void Session::addSymbol(Symbol* sym, const Item& value)
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   _p->m_version++;
   if( iter != _p->m_symbols.end() )
   {
      sym->incref();
      _p->m_symbols.insert(std::make_pair(sym, value));
   }
   else {
      iter->second = value;
   }
   _p->m_mtxSym.unlock();
}


bool Session::removeSymbol(Symbol* sym)
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   if( iter != _p->m_symbols.end() )
   {
      _p->m_version++;
      _p->m_symbols.erase(iter);
      _p->m_mtxSym.unlock();

      sym->decref();
      return true;
   }
   else {
      _p->m_mtxSym.unlock();
   }

   return false;
}


void Session::record( VMContext* ctx )
{
   _p->m_mtxSym.lock();
   _p->m_version++;
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.begin() )
   {
      Symbol* sym = iter->first;
      Item* item = ctx->resolveSymbol(sym, false);
      if( item != 0 )
      {
         iter->second.copyFromRemote(*item);
      }
      ++iter;
   }
   _p->m_mtxSym.unlock();
}


void Session::apply( VMContext* ctx ) const
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.begin() )
   {
      Symbol* sym = iter->first;
      Item* item = ctx->resolveSymbol(sym, true);
      fassert( *item != 0 );
      item->copyFromLocal(iter->second);
      ++iter;
   }
   _p->m_mtxSym.unlock();
}


void Session::store(VMContext* ctx, Storer* storer) const
{
   Class* cls = 0;
   void* data = 0;

   // the lock here is a bit wide, but having a look into Storer::store, it seems harmless.
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.begin() )
   {
      Symbol* sym = iter->first;
      storer->store(ctx, sym->handler(), sym, false);
      iter->second.forceClassInst(cls,data);
      // the data is in GC, as we know we have locked it.
      storer->store(ctx, cls, data, true);
      ++iter;
   }
   _p->m_mtxSym.unlock();

   // write a nil as end marker
   Item item;
   item.forceClassInst(cls, data);
   storer->store(ctx, cls, data, false);
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
      Item value(handler, data);
      value.deuser();
      addSymbol( sym, value );
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

}

/* end of session.cpp */
