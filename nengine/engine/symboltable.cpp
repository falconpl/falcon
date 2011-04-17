/*
   FALCON - The Falcon Programming Language.
   FILE: synmboltable.cpp

   Symbol table -- where to store local or global symbols.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 15:56:50 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/symboltable.h>
#include <falcon/string.h>

#include <vector>
#include <map>

#include "falcon/localsymbol.h"

namespace Falcon
{

class SymbolTable::Private
{
private:
   friend class SymbolTable;

   //TODO: Use our old property table?
   // Anyhow, should be optimized a bit.
   typedef std::map<String, Symbol*> SymbolMap;
   SymbolMap m_symtab;

   typedef std::vector<LocalSymbol*> SymbolVector;
   SymbolVector m_locals;

};

SymbolTable::SymbolTable()
{
   _p = new Private;
}

SymbolTable::~SymbolTable()
{
   Private::SymbolMap::iterator iter = _p->m_symtab.begin();
   while( iter != _p->m_symtab.end() )
   {
      delete iter->second;
      ++iter;
   }
}

int32 SymbolTable::localCount() const
{
   return _p->m_locals.size();
}


Symbol* SymbolTable::findSymbol( const String& name ) const
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( name );
   if( iter != _p->m_symtab.end() )
   {
      return iter->second;
   }
   return 0;
}


LocalSymbol* SymbolTable::getLocal( int32 id ) const
{
   if ( id < 0 || ((size_t)id) >= _p->m_locals.size() )
   {
      return 0;
   }
   
   return _p->m_locals[id];
}

   
Symbol* SymbolTable::addLocal( const String& name )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( name );
   if( iter != _p->m_symtab.end() )
   {
      return iter->second;
   }
   
   LocalSymbol* ls = new LocalSymbol( name, _p->m_locals.size() );
   _p->m_locals.push_back( ls );
   _p->m_symtab[name] = ls;
   
   return ls;
}

  
bool SymbolTable::addLocal( LocalSymbol* sym )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( sym->name() );
   if( iter != _p->m_symtab.end() )
   {
      return false;
   }

   sym->m_id = _p->m_locals.size();
   _p->m_locals.push_back( sym );
   _p->m_symtab[sym->name()] = sym;

   return true;
}

  
bool SymbolTable::addSymbol( Symbol* sym )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( sym->name() );
   if( iter != _p->m_symtab.end() )
   {
      return false;
   }

   _p->m_symtab[sym->name()] = sym;
   return true;
}

}

/* end of symboltable.cpp */
