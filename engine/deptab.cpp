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
   Map( &traits::t_stringptr(), &traits::t_voidp() )
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

      s = endianInt32( name->id() );
      out->write( &s, sizeof( s ) );

      s = endianInt32( data->moduleName()->id() );
      out->write( &s, sizeof( s ) );

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

      in->read( &id, sizeof( id ) );
      const String *alias = mod->getString( endianInt32( id ) );
      if( alias == 0 )
         return false;

      in->read( &id, sizeof( id ) );
      const String *modname = mod->getString( endianInt32( id ) );
      if( modname == 0 )
         return false;

      in->read( &id, sizeof( id ) );
      bool isPrivate = endianInt32(id) != 0;

      in->read( &id, sizeof( id ) );
      bool isFile = endianInt32(id) != 0;

      insert( alias, new ModuleDepData( modname, isPrivate, isFile ) );
      --s;
   }

   return true;
}


void DependTable::addDependency( const String *alias, const String *name, bool bPrivate )
{
   insert( alias, new ModuleDepData( name, bPrivate ) );
}


}

/* end of deptab.cpp */
