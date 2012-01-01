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

#define SRC "engine/symboltable.cpp"

#include <falcon/symboltable.h>
#include <falcon/symbol.h>
#include <falcon/string.h>

#include <vector>
#include <map>

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

   typedef std::vector<Symbol*> SymbolVector;
   SymbolVector m_locals;
   SymbolVector m_closed;

   Private() {}
   
   // No deep copy involved...
   Private( const Private& other ):
      m_symtab( other.m_symtab ),
      m_locals( other.m_locals ),
      m_closed( other.m_closed )
   {}
   
   ~Private() {}
};

SymbolTable::SymbolTable()
{
   _p = new Private;
}

SymbolTable::SymbolTable( const SymbolTable& other)
{
   _p = new Private( *other._p );
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

int32 SymbolTable::closedCount() const
{
   return _p->m_closed.size();
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


Symbol* SymbolTable::getLocal( int32 id ) const
{
   if ( id < 0 || ((size_t)id) >= _p->m_locals.size() )
   {
      return 0;
   }
   
   return _p->m_locals[id];
}

Symbol* SymbolTable::getClosed( int32 id ) const
{
   if ( id < 0 || ((size_t)id) >= _p->m_closed.size() )
   {
      return 0;
   }
   
   return _p->m_closed[id];
}


   
Symbol* SymbolTable::addLocal( const String& name )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( name );
   if( iter != _p->m_symtab.end() )
   {
      return iter->second;
   }
   
   Symbol* ls = new Symbol( name, Symbol::e_st_local, _p->m_locals.size() );
   _p->m_locals.push_back( ls );
   _p->m_symtab[name] = ls;
   
   return ls;
}

  
bool SymbolTable::addLocal( Symbol* sym )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( sym->name() );
   if( iter != _p->m_symtab.end() )
   {
      return false;
   }

   sym->define( Symbol::e_st_local, _p->m_locals.size() );
   _p->m_locals.push_back( sym );
   _p->m_symtab[sym->name()] = sym;

   return true;
}


   
Symbol* SymbolTable::addClosed( const String& name )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( name );
   if( iter != _p->m_symtab.end() )
   {
      return iter->second;
   }
   
   Symbol* ls = new Symbol( name, Symbol::e_st_closed, _p->m_closed.size() );
   _p->m_closed.push_back( ls );
   _p->m_symtab[name] = ls;
   
   return ls;
}

  
bool SymbolTable::addClosed( Symbol* sym )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( sym->name() );
   if( iter != _p->m_symtab.end() )
   {
      return false;
   }

   sym->define( Symbol::e_st_closed, _p->m_closed.size() );
   _p->m_closed.push_back( sym );
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
