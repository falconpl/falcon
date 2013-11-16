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

void Session::addSymbol(Symbol* sym)
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.find(sym);
   if( iter != _p->m_symbols.end() )
   {
      sym->incref();
      _p->m_symbols.insert(std::make_pair(sym, Item()));
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

void Session::record( Module* mod )
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.begin() )
   {
      Symbol* sym = iter->first;
      Item* item = mod->resolveLocally(sym);
      if( item != 0 )
      {
         iter->second = *item;
      }
      ++iter;
   }
   _p->m_mtxSym.unlock();
}

void Session::apply( Module* mod ) const
{
   _p->m_mtxSym.lock();
   Private::SymbolSet::iterator iter = _p->m_symbols.begin();
   while( iter != _p->m_symbols.begin() )
   {
      Symbol* sym = iter->first;
      Item* item = mod->resolveLocally(sym);
      if( item != 0 )
      {
         iter->second = mod->resolveLocally(sym);
      }
      ++iter;
   }
   _p->m_mtxSym.unlock();
}

}

/* end of session.cpp */
