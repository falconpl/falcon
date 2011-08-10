/*
   FALCON - The Falcon Programming Language.
   FILE: symbolmap.cpp

   A simple class orderly guarding symbols and the module they come from.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 09 Aug 2011 00:43:47 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/symbolmap.h>
#include <falcon/symbol.h>
#include <falcon/mt.h>

#include "symbolmap_private.h"

#include <map>

namespace Falcon 
{

SymbolMap::SymbolMap():
   _p( new Private )
{}

SymbolMap::~SymbolMap()
{
  delete _p; 
}
   
bool SymbolMap::add( Symbol* sym, Module* mod )
{
   Entry* e = new Entry( sym, mod );
   _p->m_mtx.lock();
   if ( _p->m_syms.find( sym->name() ) != _p->m_syms.end() )
   {
      _p->m_mtx.unlock();
      return false;
   }
   _p->m_syms[sym->name()] = e;
   _p->m_mtx.unlock();
   return true;
}

void SymbolMap::remove( const String& symName )
{
   _p->m_mtx.lock();
   Private::SymModMap::iterator pos = _p->m_syms.find(symName);
   if( pos != _p->m_syms.end() )
   {
      Entry* e = pos->second; 
      _p->m_syms.erase( pos );
      _p->m_mtx.unlock();
      delete e;
   }
   else
   {
      _p->m_mtx.unlock();
   }
}


SymbolMap::Entry* SymbolMap::find( const String& symName ) const
{
   _p->m_mtx.lock();
   Private::SymModMap::iterator pos = _p->m_syms.find(symName);
   if( pos != _p->m_syms.end() )
   {
      Entry* e = pos->second; 
      _p->m_mtx.unlock();
      return e;
   }
   
   _p->m_mtx.unlock();   
   return 0;
}


void SymbolMap::enumerate( SymbolMap::EntryEnumerator& rator ) const
{
   _p->m_mtx.lock();
   Private::SymModMap::iterator iter = _p->m_syms.begin();
   while( iter != _p->m_syms.end() )
   {
      Entry* e = iter->second;
      ++iter;
      
      _p->m_mtx.unlock();      
      rator(e);
      _p->m_mtx.lock();      
   }
   _p->m_mtx.unlock();
}

}

/* end of symbolmap.cpp */
