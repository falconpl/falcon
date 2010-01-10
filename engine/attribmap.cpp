/*
   FALCON - The Falcon Programming Language.
   FILE: attribmap.cpp

   Attribute Map - specialized string - vardef map.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jul 2009 20:42:43 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/attribmap.h>
#include <falcon/vardef.h>
#include <falcon/traits.h>
#include <falcon/string.h>
#include <falcon/stream.h>

namespace Falcon {


AttribMap::AttribMap():
   Map( &traits::t_string(), &traits::t_voidp() )
{}


AttribMap::AttribMap( const AttribMap& other ):
   Map( &traits::t_string(), &traits::t_voidp() )
{
   MapIterator iter = other.begin();
   while( iter.hasCurrent() )
   {
      insertAttrib( *(String *) iter.currentKey(),  *(VarDef**) iter.currentValue() );
      iter.next();
   }
}


AttribMap::~AttribMap()
{
   MapIterator iter = begin();
   while( iter.hasCurrent() )
   {
      delete *(VarDef**) iter.currentValue();
      iter.next();
   }
}

void AttribMap::insertAttrib( const String& name, VarDef* vd )
{
   void *found = find( &name );
   if ( found ) {
      VarDef* old = *(VarDef**) found;
      *old = *vd;
      delete vd;
   }
   else
      insert( &name, vd );
}

VarDef* AttribMap::findAttrib( const String& name )
{
   void *found = find( &name );
   if ( ! found )
      return 0;

   return *(VarDef**) found;
}


bool AttribMap::save( const Module* mod, Stream *out ) const
{
   uint32 sz = endianInt32( size() );
   out->write( &sz, sizeof( sz ) );

   MapIterator iter = begin();
   while( iter.hasCurrent() )
   {
      String* key = (String*) iter.currentKey();
      VarDef* value = *(VarDef**) iter.currentValue();

      key->serialize( out );
      value->save( mod, out );

      iter.next();
   }

   return out->good();
}


bool AttribMap::load( const Module* mod, Stream *in )
{
   uint32 sz;
   in->read( &sz, sizeof( sz ) );
   sz = endianInt32( sz );

   for( uint32 i = 0; i < sz; ++i )
   {
      String k;

      VarDef* temp = new VarDef;
      if ( ! k.deserialize( in ) || ! temp->load( mod, in ) )
      {
         delete temp;
         return false;
      }

      insertAttrib( k, temp );
   }

   return true;
}

}


/* end of attribmap.cpp */
