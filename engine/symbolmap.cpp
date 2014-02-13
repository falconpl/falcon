/*
   FALCON - The Falcon Programming Language.
   FILE: symbolmap.cpp

   Map holding local and global variable tables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 29 Dec 2012 10:03:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/varmap.cpp"

#include <falcon/string.h>
#include <falcon/symbolmap.h>
#include <falcon/symbol.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <map>
#include <vector>

namespace Falcon {

class SymbolMap::Private
{
public:

   typedef std::map<const Symbol*, int32> Map;
   typedef std::vector<const Symbol*> List;

   Map m_map;
   List m_list;

   void copy( const Private& other )
   {
      m_map.clear();
      m_list.clear();


      const List& source = other.m_list;
      for( uint32 pos = 0; pos < source.size(); ++pos )
      {
         const Symbol* sym = source[pos];
         m_map.insert( std::make_pair(sym, pos) );
         m_list.push_back(sym);
         sym->incref();
      }
   }

   ~Private()
   {
      List& source = m_list;
      List::iterator iter = source.begin();
      List::iterator end = source.end();
      while( iter != end )
      {
         const Symbol* sym = *iter;
         sym->decref();
         ++iter;
      }
   }

};

SymbolMap::SymbolMap()
{
   _p = new Private;
}

SymbolMap::SymbolMap( const SymbolMap& other )
{
   _p = new Private;
   _p->copy( *other._p );
}


SymbolMap::~SymbolMap()
{
   delete _p;
}


int32 SymbolMap::insert( const String& name )
{
   const Symbol* sym = Engine::getSymbol(name);

   Private::Map::iterator pos = _p->m_map.find( sym );
   if( pos != _p->m_map.end() ) {
      sym->decref();
      return -1;
   }
   uint32 id = _p->m_list.size();
   _p->m_map.insert( std::make_pair(sym, id) );
   _p->m_list.push_back( sym );

   return id;
}


int32 SymbolMap::insert( const Symbol* sym )
{
   Private::Map::iterator pos = _p->m_map.find( sym );
   if( pos != _p->m_map.end() ) {
      return -1;
   }
   uint32 id = _p->m_list.size();
   _p->m_map.insert( std::make_pair(sym, id) );
   _p->m_list.push_back( sym );
   sym->incref();

   return id;
}


int32 SymbolMap::find( const String& name ) const
{
   const Symbol* sym = Engine::getSymbol(name);

   Private::Map::iterator pos = _p->m_map.find( sym );
   if( pos == _p->m_map.end() ) {
      sym->decref();
      return -1;
   }
   sym->decref();
   return pos->second;
}


int32 SymbolMap::find( const Symbol*sym ) const
{
   Private::Map::iterator pos = _p->m_map.find( sym );
   if( pos == _p->m_map.end() ) {
      return -1;
   }
   return pos->second;
}


const String& SymbolMap::getNameById( uint32 id ) const
{
   static String nothing = "";
   if( id >= _p->m_list.size() ) {
      return nothing;
   }
   return _p->m_list[id]->name();
}


const Symbol* SymbolMap::getById( uint32 id ) const
{
   if( id >= _p->m_list.size() ) {
      return 0;
   }

   return _p->m_list[id];
}


uint32 SymbolMap::size() const
{
   return _p->m_list.size();
}


void SymbolMap::enumerate( Enumerator& e )
{
   Private::Map::iterator pos = _p->m_map.begin();
   Private::Map::iterator end = _p->m_map.end();

   while( pos != end )
   {
      e( pos->first->name() );
      ++pos;
   }
}

void SymbolMap::store( DataWriter* dw ) const
{
   // write JUST the size of all vectors;
   // we'll rebuild them using the variable map data.
   dw->write( (uint32) _p->m_list.size() );
   Private::List::const_iterator vmi = _p->m_list.begin();
   while( vmi != _p->m_list.end() )
   {
      const Symbol*sym = *vmi;
      dw->write( sym->name() );
      ++vmi;
   }
}


void SymbolMap::restore( DataReader* dr )
{
   uint32 size;
   dr->read(size);
   _p->m_list.reserve( size );
   for( uint32 i = 0; i < size; ++i )
   {
      String name;

      dr->read(name);
      const Symbol* sym = Engine::getSymbol(name);
      _p->m_list.push_back( sym );
      _p->m_map[sym] = i;
   }
}

}

/* end of symbolmap.cpp */
