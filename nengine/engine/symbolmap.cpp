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

#include <map>

namespace Falcon 
{

class SymbolMap::Private
{
public:
   typedef std::map<String, SymbolMap::Entry*> SymModMap;
   SymModMap m_syms;

   Private() {}
   ~Private() 
   {
      SymModMap::iterator iter = m_syms.begin();
      while( iter != m_syms.end() )
      {
         delete iter->second;
         ++iter;
      }
   }
};


SymbolMap::SymbolMap():
   _p( new Private )
{}

SymbolMap::~SymbolMap()
{
  delete _p; 
}
   
void SymbolMap::add( Symbol* sym, Module* mod )
{
   _p->m_syms[sym->name()] = new Entry( sym, mod );
}

void SymbolMap::remove( const String& symName )
{
   Private::SymModMap::iterator pos = _p->m_syms.find(symName);
   if( pos != _p->m_syms.end() )
   {
      delete pos->second;
      _p->m_syms.erase( pos );
   }
}


SymbolMap::Entry* SymbolMap::find( const String& symName ) const
{
   Private::SymModMap::iterator pos = _p->m_syms.find(symName);
   if( pos != _p->m_syms.end() )
   {
      return pos->second;
   }
   
   return 0;
}


void SymbolMap::enumerate( SymbolMap::EntryEnumerator& rator ) const
{
   Private::SymModMap::iterator iter = _p->m_syms.begin();
   while( iter != _p->m_syms.end() )
   {
      rator(iter->second);
      ++iter;
   }
}

}

/* end of symbolmap.cpp */
