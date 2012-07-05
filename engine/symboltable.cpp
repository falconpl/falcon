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
#include <falcon/syntree.h>
#include <falcon/function.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>

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

SymbolTable::SymbolTable():
   m_isEta(false)
{  
   m_localCount = 0;
   m_closedCount = 0;
   _p = new Private;
}

SymbolTable::SymbolTable( const SymbolTable& other)
{
   m_isEta = other.m_isEta;
   m_localCount = 0;
   m_closedCount = 0;
   _p = new Private( *other._p );
}

SymbolTable::~SymbolTable()
{
}


void SymbolTable::gcMark( uint32 mark ) 
{ 
   Private::SymbolMap::iterator iter = _p->m_symtab.begin();
   while( iter != _p->m_symtab.end() )
   {
      iter->second->gcMark( mark );
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


   
Symbol* SymbolTable::addLocal( const String& name, int32 line )
{
   Collector* coll = Engine::instance()->collector();
   Class* symClass = Engine::instance()->symbolClass();
   
   Private::SymbolMap::iterator iter = _p->m_symtab.find( name );
   if( iter != _p->m_symtab.end() )
   {
      return iter->second;
   }
   
   Symbol* ls = new Symbol( name, 
      Symbol::e_st_local, _p->m_locals.size(), line );
   FALCON_GC_STORE(coll, symClass, ls );
   _p->m_locals.push_back( ls );
   _p->m_symtab[name] = ls;
   
   return ls;
}

   
Symbol* SymbolTable::addClosed( const String& name, int32 line )
{
   Collector* coll = Engine::instance()->collector();
   Class* symClass = Engine::instance()->symbolClass();
   
   Private::SymbolMap::iterator iter = _p->m_symtab.find( name );
   if( iter != _p->m_symtab.end() )
   {
      return iter->second;
   }
   
   Symbol* ls = new Symbol( name, 
      Symbol::e_st_closed, _p->m_closed.size(), line );
   
   FALCON_GC_STORE(coll, symClass, ls );
   _p->m_closed.push_back( ls );
   _p->m_symtab[name] = ls;
   
   return ls;
}


void SymbolTable::store( DataWriter* dw )
{
   dw->write( (uint32) _p->m_locals.size() );
   dw->write( (uint32) _p->m_closed.size() );
   
}


void SymbolTable::restore( DataReader* dr )
{
   dr->read( m_localCount );
   dr->read( m_closedCount );
}

void SymbolTable::flatten( ItemArray& array )
{
   Class* symClass = Engine::instance()->symbolClass();   
   
   array.reserve( _p->m_locals.size() + _p->m_closed.size() );   
   {
      Private::SymbolVector::iterator iter = _p->m_locals.begin();
      while( iter != _p->m_locals.end() ) {
         Symbol* sym = *iter;
         array.append( Item( symClass, sym ) );
         ++iter;
      }
   }
      
   {
      Private::SymbolVector::iterator iter = _p->m_closed.begin();
      while( iter != _p->m_closed.end() ) {
         Symbol* sym = *iter;
         array.append( Item( symClass, sym ) );
         ++iter;
      }
   }
}


void SymbolTable::unflatten( ItemArray& array )
{
   uint32 pos = 0;   
   fassert( array.length() >= m_localCount+m_closedCount );
   
   // first the locals
   while( pos < m_localCount )
   {
      Symbol* symbol = static_cast<Symbol*>(array[pos++].asInst());
      _p->m_locals.push_back( symbol );
      _p->m_symtab[symbol->name()] = symbol;
   }
   
   // then the closed the locals
   while( pos < m_localCount+m_closedCount )
   {
      Symbol* symbol = static_cast<Symbol*>(array[pos++].asInst());
      _p->m_closed.push_back( symbol );
      _p->m_symtab[symbol->name()] = symbol;
   }
   
}

}

/* end of symboltable.cpp */
