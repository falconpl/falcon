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
#include <falcon/.h>

#include <map>

namespace Falcon {

class Session::Private
{
public:
   typedef std::map<Symbol*, Item> SymbolSet;

   Mutex m_mtxSym;
   SymbolSet m_symbols;

   Private() {}
   ~Private() {}
};

Session::Session()
{
   _p = new Private;
}

Session::Session( const String& name ):
         m_name(name)
{
   _p = new Private;
}

Session::~Session()
{
   delete _p;
}


void Session::addSymbol(Symbol* sym, const Item& value)
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   if( iter != _p->m_symbols.end() )
   {
      sym->incref();
      _p->m_symbols.insert(std::make_pair(sym, value));
   }
   _p->m_mtxSym.unlock();
}


bool Session::removeSymbol(Symbol* sym)
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   if( iter != _p->m_symbols.end() )
   {
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
      if( handler->typeID() == FLC_T)
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

}

/* end of session.cpp */
