/*
   FALCON - The Falcon Programming Language.
   FILE: deptab.cpp

   Dependency table support for modules.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 21 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Dependency table support for modules.
*/

#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/deptab.h>
#include <falcon/common.h>
#include <falcon/module.h>

namespace Falcon {

DependTable::DependTable():
   Map( &traits::t_stringptr_own(), &traits::t_voidp() )
{
}

DependTable::~DependTable()
{
   MapIterator iter = begin();
   while( iter.hasCurrent() ) {
      delete *(ModuleDepData **) iter.currentValue();
      iter.next();
   }
}

bool DependTable::save( Stream *out ) const
{
   uint32 s = endianInt32(size());
   out->write( &s, sizeof( s ) );

   MapIterator iter = begin();
   while( iter.hasCurrent() ) {
      const String *name = *(const String **) iter.currentKey();
      const ModuleDepData *data = *(const ModuleDepData **) iter.currentValue();

      name->serialize( out );
      data->moduleName().serialize( out );

      s = endianInt32( data->isPrivate() ? 1 : 0 );
      out->write( &s, sizeof( s ) );

      s = endianInt32( data->isFile() ? 1 : 0 );
      out->write( &s, sizeof( s ) );

      iter.next();
   }
   return true;
}

bool DependTable::load( Module *mod, Stream *in )
{
   uint32 s;
   in->read( &s, sizeof( s ) );
   s = endianInt32( s );

   while( s > 0 )
   {
      uint32 id;

      String alias, modname;
      if ( ! alias.deserialize( in, false )
           || ! modname.deserialize( in, false ) )
      {
         return false;
      }

      in->read( &id, sizeof( id ) );
      bool isPrivate = endianInt32(id) != 0;

      in->read( &id, sizeof( id ) );
      bool isFile = endianInt32(id) != 0;

      insert( new String(alias), new ModuleDepData( modname, isPrivate, isFile ) );
      --s;
   }

   return true;
}


void DependTable::addDependency( const String &alias, const String& name, bool bPrivate, bool bFile )
{
   insert( new String(alias), new ModuleDepData( name, bPrivate, bFile ) );
}


}

/* end of deptab.cpp */
