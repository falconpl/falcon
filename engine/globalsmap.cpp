/*
   FALCON - The Falcon Programming Language.
   FILE: globalsmap.cpp

   Map holding variables and associated data for global storage.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 08 Jan 2013 18:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/globalsmap.cpp"

#include <falcon/string.h>
#include <falcon/globalsmap.h>
#include <falcon/variable.h>
#include <falcon/module.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>
#include <falcon/symbol.h>

#include <map>
#include <vector>

namespace Falcon {

//================================================================
// VarDataMap::Private
//================================================================

class GlobalsMap::Private
{
public:
   typedef std::map<Symbol*, Data*> VariableMap;

   VariableMap m_variables;
   VariableMap m_exports;

   uint32 m_lastGCMark;

   Private():
      m_lastGCMark(0)
   {};

   ~Private() {
      VariableMap::iterator iter = m_variables.begin();
      VariableMap::iterator end = m_variables.end();
      while( iter != end ) {
         iter->first->decref();
         delete iter->second;
         ++iter;
      }
   }

};

//================================================================
// GlobalsMap
//================================================================

GlobalsMap::GlobalsMap():
         m_bExportAll( false )
{
   _p = new Private;
}

GlobalsMap::~GlobalsMap()
{
   delete _p;
}


void GlobalsMap::gcMark( uint32 mark )
{
   if( _p->m_lastGCMark != mark )
   {
      _p->m_lastGCMark = mark;

      Private::VariableMap::iterator giter = _p->m_variables.begin();
      Private::VariableMap::iterator gend = _p->m_variables.end();

      while( giter != gend ) {
         Data* vd = giter->second;
         vd->m_data->gcMark( mark );
         ++giter;
      }
   }
}


uint32 GlobalsMap::lastGCMark() const
{
   return _p->m_lastGCMark;
}


GlobalsMap::Data* GlobalsMap::add( const String& name, const Item& value, bool bExport )
{
   Symbol* sym = Engine::getSymbol(name);
   Private::VariableMap::iterator pos = _p->m_variables.find( sym );
   if( pos != _p->m_variables.end() ) {
      // we don't have to keep the allocated symbol
      sym->decref();
     return 0;
   }

   // insert the value and save its pointer.
   Data* data = new Data(value);
   _p->m_variables[sym] = data;
   if( bExport )
   {
     _p->m_exports[sym] = data;
   }

   return data;
}


GlobalsMap::Data* GlobalsMap::add( Symbol* sym, const Item& value, bool bExport )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( sym );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }

   // need a new reference for the symbol
   sym->incref();
   // insert the value and save its pointer.
   Data* data = new Data(value);
   _p->m_variables[sym] = data;
   if( bExport )
   {
      _p->m_exports[sym] = data;
   }

   return data;
}


GlobalsMap::Data* GlobalsMap::promote( const String& name, const Item& value, bool bExport )
{
   Symbol* sym = Engine::getSymbol(name);
   Data* dt = promote( sym, value, bExport );
   sym->decref();
   return dt;
}


GlobalsMap::Data* GlobalsMap::promote( Symbol* sym, const Item& value, bool bExport )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( sym );
   Data* dt = 0;
   if( pos != _p->m_variables.end() ) {
      dt = pos->second;
      dt->m_storage = value;
      dt->m_bExtern = false;
      dt->m_data = &dt->m_storage;
   }
   else {
     dt = new Data(value);
     // need a new reference for the symbol
     sym->incref();
     _p->m_variables[sym] = dt;
     if( bExport )
     {
        _p->m_exports[sym] = dt;
     }
   }

   return dt;
}


GlobalsMap::Data* GlobalsMap::addExtern( Symbol* sym, Item* value )
{
   Data* varData;

   Private::VariableMap::iterator pos = _p->m_variables.find( sym );
   if( pos != _p->m_variables.end() ) {
      varData = pos->second;
      varData->m_data = value;
   }
   else {
      varData = new Data(value);
      // need a new reference for the symbol
      sym->incref();
      // insert the value and save its pointer.
      _p->m_variables[sym] = varData;
   }

   varData->m_bExtern = true;

   return varData;
}


GlobalsMap::Data* GlobalsMap::addExtern( const String& symName, Item* value )
{
   Symbol* sym = Engine::getSymbol(symName);
   Data* dt = addExtern( sym, value );
   sym->decref();
   return dt;
}


bool GlobalsMap::remove( const String& name )
{
   Symbol* sym = Engine::getSymbol(name);
   bool result = remove(sym);
   sym->decref();
   return result;
}

GlobalsMap::Data* GlobalsMap::exportGlobal( const String& name, bool &bAlready )
{
   Symbol* sym = Engine::getSymbol(name);
   Data* result = exportGlobal(sym, bAlready );
   sym->decref();
   return result;
}

GlobalsMap::Data* GlobalsMap::exportGlobal( Symbol* sym, bool &bAlready )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( sym );
   if( pos == _p->m_variables.end() ) {
      return 0;
   }

   Data* vd = pos->second;
   pos = _p->m_exports.find( sym );
   bAlready = pos != _p->m_exports.end();
   if( ! bAlready )
   {
      _p->m_exports.insert(std::make_pair(sym, vd));
   }
   return vd;
}

bool GlobalsMap::remove( Symbol* sym )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( sym );
   if( pos == _p->m_variables.end() ) {
      return false;
   }

   Data* vd = pos->second;
   _p->m_variables.erase(pos);
   _p->m_exports.erase(sym); // remove in case it's also there.
   delete vd;
   sym->decref();

   return true;
}


Item* GlobalsMap::getValue( const String& name ) const
{
   Symbol* sym = Engine::getSymbol(name);
   Private::VariableMap::iterator pos = _p->m_variables.find(sym);
   sym->decref();
   if( pos == _p->m_variables.end() ) {
      return 0;
   }

   Data* vd = pos->second;
   return vd->m_data;
}

Item* GlobalsMap::getValue( Symbol* sym ) const
{
   Private::VariableMap::iterator pos = _p->m_variables.find(sym);
   if( pos == _p->m_variables.end() ) {
      return 0;
   }

   Data* vd = pos->second;
   return vd->m_data;
}


GlobalsMap::Data* GlobalsMap::get( const String& name ) const
{
   Symbol* sym = Engine::getSymbol(name);
   Private::VariableMap::iterator pos = _p->m_variables.find(sym);
   sym->decref();
   if( pos == _p->m_variables.end() ) {
      return 0;
   }

   return pos->second;
}


GlobalsMap::Data* GlobalsMap::get( Symbol* sym ) const
{
   Private::VariableMap::iterator pos = _p->m_variables.find(sym);
   if( pos == _p->m_variables.end() ) {
      return 0;
   }

   return pos->second;
}


bool GlobalsMap::isExported( const String& name ) const
{
   Symbol* sym = Engine::getSymbol(name);
   bool result = _p->m_exports.find( sym ) != _p->m_exports.end();
   sym->decref();
   return result;
}


bool GlobalsMap::isExported( Symbol* sym ) const
{
   bool result = _p->m_exports.find( sym ) != _p->m_exports.end();
   return result;
}


void GlobalsMap::enumerateExports( VariableEnumerator& rator ) const
{
   Private::VariableMap::iterator iter, end;

   if( m_bExportAll ) {
      iter = _p->m_variables.begin();
      end = _p->m_variables.end();
   }
   else {
      iter = _p->m_exports.begin();
      end = _p->m_exports.end();
   }

   while( iter != end ) {
      Symbol* sym = iter->first;
      if( ! sym->name().empty() && sym->name().getCharAt(0) != '_' )
      {
         Data* vd = iter->second;
         rator( iter->first, vd->m_data );
      }

      ++iter;
   }
}


void GlobalsMap::enumerate( VariableEnumerator& rator ) const
{
   Private::VariableMap::iterator iter, end;

   iter = _p->m_variables.begin();
   end = _p->m_variables.end();

   while( iter != end ) {
      Symbol* sym = iter->first;
      if( ! sym->name().empty() )
      {
         Data* vd = iter->second;
         rator( sym, vd->m_data );
      }

      ++iter;
   }
}


uint32 GlobalsMap::size() const
{
   return _p->m_variables.size();
}


void GlobalsMap::flatten( VMContext*, ItemArray& subItems ) const
{
   Private::VariableMap::iterator iter = _p->m_variables.begin();
   Private::VariableMap::iterator end = _p->m_variables.end();

   subItems.reserve(_p->m_variables.size() * 3);
   while( iter != end )
   {
      Symbol* sym = iter->first;
      Data* vd = iter->second;
      subItems.append( sym );
      // ignore the data where the global points; store externals as nil
      // we'll store the locally stored globals only.
      subItems.append( vd->m_storage );

      int64 flags = (isExported(sym) ? 1 : 0) + (vd->m_bExtern ? 2:0);
      subItems.append( flags );

      ++iter;
   }
}


void GlobalsMap::unflatten( VMContext*, ItemArray& subItems, uint32 start, uint32 &count )
{
   uint32 c = start;
   while( c+3 <= subItems.length() && ! subItems[c].isNil() )
   {
      Symbol* sym = static_cast<Symbol*>(subItems[c++].asInst());
      Data* vd = new Data;
      vd->m_storage = subItems[c++];
      vd->m_data = &vd->m_storage;

      int64 flags = subItems[c++].asInteger();
      vd->m_bExtern = (flags & 2) != 0;
      _p->m_variables[sym] = vd;
      if( (flags & 1) != 0 )
      {
         _p->m_exports[sym] = vd;
      }
   }

   count = c;
}

}

/* end of globalsmap.cpp */
