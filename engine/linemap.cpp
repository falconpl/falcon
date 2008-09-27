/*
   FALCON - The Falcon Programming Language.
   FILE: linemap.cpp

   Line map with module serialization support.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 21 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Line map with module serialization support.
*/

#include <falcon/linemap.h>
#include <falcon/stream.h>
#include <falcon/common.h>
#include <falcon/traits.h>

namespace Falcon {

LineMap::LineMap():
   Map( &traits::t_int(), &traits::t_int() )
{}

bool LineMap::save( Stream *out ) const
{
   uint32 s = endianInt32(size());
   out->write(  &s , sizeof( s ) );
   MapIterator iter = begin();
   while( iter.hasCurrent() )
   {

      s = endianInt32( *(uint32 *) iter.currentKey() );
      out->write(  &s , sizeof( s ) );
      s = endianInt32( *(uint32 *) iter.currentValue() );
      out->write(  &s , sizeof( s ) );
      iter.next();
   }

   return true;
}

bool LineMap::load( Stream *in )
{
   uint32 s;
   in->read(  &s , sizeof( s ) );
   s = endianInt32( s );

   while( s > 0 )
   {
      uint32 pos;
      uint32 line;
      in->read(  &pos , sizeof( pos ) );
      pos = endianInt32( pos );
      in->read(  &line , sizeof( line ) );
      line = endianInt32( line );
      insert( &pos, &line );
      --s;
   }

   return true;
}

}


/* end of linemap.cpp */
