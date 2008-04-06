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


bool DependTable::save( Stream *out ) const
{
   uint32 s = endianInt32(size());
   out->write( &s, sizeof( s ) );

   ListElement *iter = begin();
   while( iter != 0 ) {
      const String *str = (const String *) iter->data();
      s = endianInt32( str->id() );
      out->write( &s, sizeof( s ) );
      iter = iter->next();
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
      in->read(  &id, sizeof( id ) );
      const String *str = mod->getString( endianInt32( id ) );
      if( str == 0 )
         return false;
      pushBack( str );
      --s;
   }

   return true;
}

}


/* end of deptab.cpp */
