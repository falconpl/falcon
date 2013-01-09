/*
   FALCON - The Falcon Programming Language.
   FILE: varmap.cpp

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
#include <falcon/varmap.h>
#include <falcon/variable.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <map>
#include <vector>

namespace Falcon {

class VarMap::Private
{
public:

   typedef std::map<String, Variable> VariableMap;
   typedef std::vector<const String*> NameList;

   VariableMap m_variables;
   NameList m_paramNames;
   NameList m_localNames;
   NameList m_closedNames;
   NameList m_globalNames;
   NameList m_externNames;
};

VarMap::VarMap():
     m_bEta(true)
{
   _p = new Private;
}

VarMap::~VarMap()
{
   delete _p;
}


Variable* VarMap::addParam( const String& name )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }
   uint32 id = _p->m_paramNames.size();

   pos = _p->m_variables.insert( std::make_pair(name, Variable( Variable::e_nt_param, id )) ).first;
   _p->m_paramNames.push_back(&pos->first);

   return &pos->second;

}


Variable* VarMap::addLocal( const String& name )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }
   uint32 id = _p->m_localNames.size();

   pos = _p->m_variables.insert( std::make_pair(name, Variable( Variable::e_nt_local, id )) ).first;
   _p->m_localNames.push_back(&pos->first);

   return &pos->second;
}


Variable* VarMap::addClosed( const String& name )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }
   uint32 id = _p->m_closedNames.size();

   pos = _p->m_variables.insert( std::make_pair(name, Variable( Variable::e_nt_local, id )) ).first;
   _p->m_closedNames.push_back(&pos->first);

   return &pos->second;
}

Variable* VarMap::addGlobal( const String& name )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }
   uint32 id = _p->m_globalNames.size();

   pos = _p->m_variables.insert( std::make_pair(name, Variable( Variable::e_nt_local, id )) ).first;
   _p->m_globalNames.push_back(&pos->first);

   return &pos->second;
}

Variable* VarMap::addExtern( const String& name )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }
   uint32 id = _p->m_externNames.size();

   pos = _p->m_variables.insert( std::make_pair(name, Variable( Variable::e_nt_local, id )) ).first;
   _p->m_externNames.push_back(&pos->first);

   return &pos->second;
}


Variable* VarMap::find( const String& name ) const
{
   Private::VariableMap::iterator pos = _p->m_variables.find( name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }

   return &pos->second;
}


const String& VarMap::getParamName( uint32 id ) const
{
   return *_p->m_paramNames[id];
}

const String& VarMap::getLoacalName( uint32 id ) const
{
   return *_p->m_localNames[id];
}

const String& VarMap::getClosedName( uint32 id ) const
{
   return *_p->m_closedNames[id];
}

const String& VarMap::getGlobalName( uint32 id ) const
{
   return *_p->m_globalNames[id];
}

const String& VarMap::getExternName( uint32 id ) const
{
   return *_p->m_externNames[id];
}

uint32 VarMap::paramCount() const
{
   return _p->m_paramNames.size();
}

uint32 VarMap::localCount() const
{
   return _p->m_localNames.size();
}

uint32 VarMap::closedCount() const
{
   return _p->m_closedNames.size();
}

uint32 VarMap::globalCount() const
{
   return _p->m_globalNames.size();
}

uint32 VarMap::externCount() const
{
   return _p->m_externNames.size();
}

uint32 VarMap::allLocalCount() const
{
   return _p->m_paramNames.size() + _p->m_localNames.size();
}

void VarMap::enumerate( Enumerator& e )
{
   Private::VariableMap::iterator pos = _p->m_variables.begin();
   Private::VariableMap::iterator end = _p->m_variables.end();

   while( pos != end ) {
      e( pos->first, pos->second );
      ++pos;
   }
}

void VarMap::store( DataWriter* dw ) const
{
   // write JUST the size of all vectors;
   // we'll rebuild them using the variable map data.
   dw->write( isEta() );
   dw->write( (uint32) _p->m_paramNames.size() );
   dw->write( (uint32) _p->m_localNames.size() );
   dw->write( (uint32) _p->m_closedNames.size() );
   dw->write( (uint32) _p->m_globalNames.size() );
   dw->write( (uint32) _p->m_externNames.size() );


   dw->write( (uint32) _p->m_variables.size() );
   Private::VariableMap::const_iterator vmi = _p->m_variables.begin();
   while( vmi != _p->m_variables.end() )
   {
      const String& name = vmi->first;
      const Variable& var = vmi->second;

      dw->write( name );

      dw->write( (char) var.type() );
      dw->write( var.declaredAt() );
      dw->write( var.id() );
      dw->write( var.isConst() );

      ++vmi;
   }
}


void VarMap::restore( DataReader* dr )
{
   bool ie;
   dr->read(ie);
   setEta(ie);

   uint32 size;
   dr->read(size);
   _p->m_paramNames.resize( size, 0 );
   dr->read(size);
   _p->m_localNames.resize( size, 0 );
   dr->read(size);
   _p->m_closedNames.resize( size, 0 );
   dr->read(size);
   _p->m_globalNames.resize( size, 0 );
   dr->read(size);
   _p->m_externNames.resize( size, 0 );

   dr->read(size);
   for( uint32 i = 0; i < size; ++i )
   {
      String name;

      dr->read(name);

      char type;
      int declaredAt;
      uint32 id;
      bool isConst;

      dr->read( type );
      dr->read( declaredAt );
      dr->read( id );
      dr->read( isConst );
      Variable::type_t tt = (Variable::type_t) type;

      const String* sval = &(_p->m_variables.insert( std::make_pair( name,
               Variable( tt, id, declaredAt, isConst)
               ) ).first)->first;

      switch( tt ) {
      case Variable::e_nt_local: _p->m_localNames[id] = sval; break;
      case Variable::e_nt_param: _p->m_paramNames[id] = sval; break;
      case Variable::e_nt_global: _p->m_globalNames[id] = sval; break;
      case Variable::e_nt_closed: _p->m_closedNames[id] = sval; break;
      case Variable::e_nt_extern: _p->m_externNames[id] = sval; break;
      case Variable::e_nt_undefined: break;
      }
   }
}

}

/* end of varmap.cpp */
