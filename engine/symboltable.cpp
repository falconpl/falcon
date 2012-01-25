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
   m_ownedby( e_owned_none )
{
   m_owner.function = 0;
   _p = new Private;
}

SymbolTable::SymbolTable( Function* parent ):
   m_ownedby( e_owned_function )
{
   m_owner.function = parent;
   _p = new Private;
}

SymbolTable::SymbolTable( SynTree* parent ):
   m_ownedby( e_owned_syntree )
{
   m_owner.syntree = parent;
   _p = new Private;
}

SymbolTable::SymbolTable( Function* parent, const SymbolTable& other):
   m_ownedby( e_owned_function )
{
    m_owner.function = parent;
   _p = new Private( *other._p );
}

SymbolTable::SymbolTable( SynTree* parent, const SymbolTable& other):
   m_ownedby( e_owned_syntree )
{
   m_owner.syntree = parent;
   _p = new Private( *other._p );
}

SymbolTable::SymbolTable( const SymbolTable& other):
   m_ownedby( e_owned_none )
{
   m_owner.function = 0;
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

void SymbolTable::gcMark( uint32 mark ) 
{ 
   if( m_ownedby == e_owned_function )
   {
      m_owner.function->gcMark( mark );
   }
   else if ( m_ownedby == e_owned_syntree ) 
   {
      m_owner.syntree->gcMark( mark );
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
   
   Symbol* ls = new Symbol( name, this, _p->m_locals.size() );
   _p->m_locals.push_back( ls );
   _p->m_symtab[name] = ls;
   
   return ls;
}

   
Symbol* SymbolTable::addClosed( const String& name )
{
   Private::SymbolMap::iterator iter = _p->m_symtab.find( name );
   if( iter != _p->m_symtab.end() )
   {
      return iter->second;
   }
   
   Symbol* ls = Symbol::ClosedSymbol( name, this, _p->m_closed.size() );
   _p->m_closed.push_back( ls );
   _p->m_symtab[name] = ls;
   
   return ls;
}


void SymbolTable::store( DataWriter* dw )
{
   dw->write( (size_t) _p->m_locals.size() );
   {
      Private::SymbolVector::iterator iter = _p->m_locals.begin();
      while( iter != _p->m_locals.end() ) {
         Symbol* sym = *iter;
         dw->write( sym->name() );
         ++iter;
      }
   }
   
   dw->write( (size_t) _p->m_closed.size() );
   {
      Private::SymbolVector::iterator iter = _p->m_closed.begin();
      while( iter != _p->m_closed.end() ) {
         Symbol* sym = *iter;
         dw->write( sym->name() );
         ++iter;
      }
   }
}


void SymbolTable::restore( DataReader* dr )
{
   size_t size;
   
   dr->read(size);
   {
      for( size_t i = 0; i < size; ++i ) {
         String name;
         dr->read(name);
         addLocal(name);
      }
   }
   
   dr->read(size);
   {
      for( size_t i = 0; i < size; ++i ) {
         String name;
         dr->read(name);
         addClosed(name);
      }
   }
}

  
}

/* end of symboltable.cpp */
