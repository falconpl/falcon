/*
   FALCON - The Falcon Programming Language.
   FILE: vardatamap.cpp

   Map holding variables and associated data for global storage.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 08 Jan 2013 18:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/vardatamap.cpp"

#include <falcon/string.h>
#include <falcon/vardatamap.h>
#include <falcon/variable.h>
#include <falcon/module.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>

#include <map>
#include <vector>

namespace Falcon {

//================================================================
// VarDataMap::Private
//================================================================

class VarDataMap::Private
{
public:

   class StringPtrCmp {
   public:
      bool operator ()( const String* p1, const String* p2 ) const {
         return *p1 < *p2;
      }
   };

   typedef std::map<const String*, VarData*, StringPtrCmp> VariableMap;
   typedef std::vector<VarData*> VariableVector;

   VariableMap m_variables;
   VariableMap m_exports;
   VariableVector m_varsByID;

   Private() {};

   ~Private() {
      VariableVector::iterator iter = m_varsByID.begin();
      VariableVector::iterator end = m_varsByID.end();
      while( iter != end ) {
         delete *iter;
         ++iter;
      }
   }

};

//================================================================
// VarDataMap
//================================================================

VarDataMap::VarDataMap():
     m_lastGCMark(0)
{
   _p = new Private;
}

VarDataMap::~VarDataMap()
{
   delete _p;
}


void VarDataMap::gcMark( uint32 mark )
{
   if( m_lastGCMark != mark )
   {
      m_lastGCMark = mark;

      // Extern come from some module space, and we just marked it.
      // we can concentrate on statics.
      Private::VariableMap::iterator giter = _p->m_variables.begin();
      Private::VariableMap::iterator gend = _p->m_variables.end();

      while( giter != gend ) {
         VarData* vd = giter->second;
         vd->m_storage.gcMark( mark );
         ++giter;
      }
   }
}


VarDataMap::VarData* VarDataMap::addGlobal( const String& name, const Item& value, bool bExport )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( &name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }

   // insert the value and save its pointer.
   uint32 id = _p->m_varsByID.size();
   Variable var(Variable::e_nt_global, id, 0, false);
   VarData* vd = new VarData( name, var, value, false, bExport );
   _p->m_variables[&vd->m_name] = vd;
   _p->m_varsByID.push_back(vd);

   if( bExport ) {
      _p->m_exports[&vd->m_name] = vd;
   }

   return vd;
}


VarDataMap::VarData* VarDataMap::addExtern( const String& name, bool bExport )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( &name );
   if( pos != _p->m_variables.end() ) {
      return 0;
   }

   // insert the value and save its pointer.
   uint32 id = _p->m_varsByID.size();
   Variable var(Variable::e_nt_extern, id, 0, false);
   VarData* vd = new VarData( name, var, 0, bExport );
   _p->m_variables[&vd->m_name] = vd;
   _p->m_varsByID.push_back(vd);

   if( bExport ) {
      _p->m_exports[&vd->m_name] = vd;
   }

   return vd;
}


bool VarDataMap::removeGlobal( const String& name )
{
   Private::VariableMap::iterator pos = _p->m_variables.find( &name );
   if( pos == _p->m_variables.end() ) {
      return false;
   }

   VarData* vd = pos->second;
   _p->m_varsByID[vd->m_var.id()] = 0;
   _p->m_variables.erase(pos);
   delete vd;

   return true;
}


bool VarDataMap::removeGlobal( uint32 id )
{
   if( id >= _p->m_varsByID.size() ) {
      return false;
   }

   VarData* vd = _p->m_varsByID[id];
   _p->m_varsByID[id] = 0;
   _p->m_variables.erase(&vd->m_name);
   delete vd;

   return true;
}


void VarDataMap::forwardNS( VarDataMap* other, const String& remoteNS, const String& localNS )
{
   Private::VariableMap& globs = other->_p->m_variables;


   // search the symbols in the remote namespace.
   String nsPrefix = remoteNS + ".";
   Private::VariableMap::iterator gi = globs.lower_bound(&nsPrefix);
   while( gi != globs.end() && gi->first->startsWith(nsPrefix) )
   {
      VarData* vdother = gi->second;

      String localName = localNS + "." + vdother->m_name.subString(nsPrefix.length());
      if( vdother->m_data != 0 ) {
         // it's ok if it returns 0; we don't want to override our variables.
         addGlobal(localName, *vdother->m_data, false );
      }

      // otherwise, the source module is not fully resolved; it should not be here.
      ++gi;
   }
}

void VarDataMap::exportNS( Module* source, const String& sourceNS, Module* target, const String& targetNS )
{

   String srcName, tgName;
   if( sourceNS.size() != 0 )
   {
      srcName = sourceNS + ".";
   }


   Private::VariableMap& srcGlobals = _p->m_variables;
   Private::VariableMap::iterator glb = srcGlobals.lower_bound( &srcName );
   while( glb != srcGlobals.end() )
   {
      const String& name = *glb->first;
      if( ! name.startsWith( srcName ) )
      {
         // we're done
         break;
      }

      // find the target name.
      tgName = targetNS.size() == 0 ? name : targetNS + "." + name;
      // import it.
      target->importValue( tgName, source, glb->second->m_data );

      ++glb;
   }
}


VarDataMap::VarData* VarDataMap::addExport( const String& name, bool &bAlready )
{
   Private::VariableMap::iterator pos = _p->m_exports.find( &name );
   if( pos != _p->m_exports.end() ) {
      bAlready = true;
   }
   else {
      bAlready = false;
      pos = _p->m_variables.find( &name );
      if( pos == _p->m_variables.end() ) {
         return 0;
      }

      VarData* vd = pos->second;
      vd->m_bExported = true;
      _p->m_exports[&vd->m_name] = vd;
   }

   return pos->second;
}


Item* VarDataMap::getGlobalValue( const String& name ) const
{
   Private::VariableMap::iterator pos = _p->m_variables.find(&name);
   if( pos == _p->m_variables.end() ) {
      return 0;
   }

   VarData* vd = pos->second;
   return vd->m_data;
}


Item* VarDataMap::getGlobalValue( uint32 id ) const
{
   if( id >= _p->m_varsByID.size() ) {
      return 0;
   }

   VarData* vd = _p->m_varsByID[id];
   return vd->m_data;
}


VarDataMap::VarData* VarDataMap::getGlobal( const String& name ) const
{
   Private::VariableMap::iterator pos = _p->m_variables.find(&name);
   if( pos == _p->m_variables.end() ) {
      return 0;
   }

   VarData* vd = pos->second;
   return vd;
}


VarDataMap::VarData* VarDataMap::getGlobal( uint32 id ) const
{
   if( id >= _p->m_varsByID.size() ) {
      return 0;
   }

   VarData* vd = _p->m_varsByID[id];
   return vd;
}


bool VarDataMap::isExported( const String& name ) const
{
   return _p->m_exports.find( &name ) != _p->m_exports.end();
}


void VarDataMap::enumerateExports( VariableEnumerator& rator ) const
{
   Private::VariableMap::iterator iter = _p->m_exports.begin();
   Private::VariableMap::iterator end = _p->m_exports.end();

   while( iter != end ) {
      VarData* vd = iter->second;
      rator( vd->m_name, vd->m_var, *vd->m_data );
      ++iter;
   }
}


bool VarDataMap::promoteExtern( uint32 id, const Item& value, int32 redeclaredAt )
{
   if( id >= _p->m_varsByID.size() ) {
      return false;
   }

   VarData* vd = _p->m_varsByID[id];
   Variable* var = &vd->m_var;

   if( var->type() != Variable::e_nt_extern ) {
      return false;
   }

   if( redeclaredAt != 0 )
   {
      var->type( Variable::e_nt_global );
   }

   vd->m_storage = value;
   vd->m_data = &vd->m_storage;

   if( redeclaredAt > 0 ) {
      var->declaredAt(redeclaredAt);
   }
   var->isResolved(true);

   return true;
}

uint32 VarDataMap::size() const
{
   return _p->m_variables.size();
}


void VarDataMap::store( DataWriter* dw ) const
{
   uint32 size = _p->m_varsByID.size();
   dw->write( size );

   Private::VariableVector::iterator iter = _p->m_varsByID.begin();
   Private::VariableVector::iterator end = _p->m_varsByID.end();

   while( iter != end )
   {
      VarData* vd = *iter;
      dw->write( vd->m_name );
      dw->write( vd->m_bExported );

      Variable& var = vd->m_var;
      dw->write( (byte) var.type() );
      dw->write( var.id() );
      dw->write( var.declaredAt() );
      dw->write( var.isConst() );

      ++iter;
   }
}

void VarDataMap::restore( DataReader* dr )
{
   uint32 size = 0;
   dr->read( size );
   _p->m_varsByID.reserve(size);

   for( uint32 i = 0; i < size; ++ i )
   {
      String name;
      bool exported;
      byte type;
      uint32 id;
      int32 declaredAt;
      bool isConst;

      dr->read( name );
      dr->read( exported );
      dr->read( type );
      dr->read( id );
      dr->read( declaredAt );
      dr->read( isConst );

      VarData* vd = new VarData( name,
               Variable( (Variable::type_t) type, id, declaredAt, isConst ),
               0, exported );

      _p->m_variables[&vd->m_name] = vd;
      _p->m_varsByID.push_back( vd );

      if( exported ) {
         _p->m_exports[&vd->m_name] = vd;
      }
   }
}


void VarDataMap::flatten( VMContext*, ItemArray& subItems ) const
{
   Private::VariableVector::iterator iter = _p->m_varsByID.begin();
   Private::VariableVector::iterator end = _p->m_varsByID.end();

   while( iter != end )
   {
      VarData* vd = *iter;
      if( vd->m_var.type() == Variable::e_nt_global )
      {
         fassert( vd->m_data != 0 );
         subItems.append( *vd->m_data );
      }
      ++iter;
   }
}


void VarDataMap::unflatten( VMContext*, ItemArray& subItems, uint32 start, uint32 &count )
{
   Private::VariableVector::iterator iter = _p->m_varsByID.begin();
   Private::VariableVector::iterator end = _p->m_varsByID.end();

   uint32 c = 0;
   while( start < subItems.length() && iter != end )
   {
      VarData* vd = *iter;
      if( vd->m_var.type() == Variable::e_nt_global )
      {
         const Item& value = subItems[start++];
         vd->m_storage = value;
         vd->m_data = &vd->m_storage;
         ++c;
      }

      ++iter;
   }

   count = c;
}

}

/* end of vardatamap.cpp */
