/*
   FALCON - The Falcon Programming Language
   FILE: garbageable.cpp

   Garbageable objects, declaration.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun gen 22 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Garbageable items definition
*/

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <falcon/vm.h>

namespace Falcon {

Garbageable::Garbageable( VMachine *vm, uint32 size ):
   m_origin( vm ),
   m_gcSize( size )
{
   vm->store( this );
}

Garbageable::Garbageable( const Garbageable &other ):
   m_origin( other.m_origin ),
   m_gcSize( 0 )
{
   other.m_origin->store( this );
}

void Garbageable::updateAllocSize( uint32 nSize )
{
   if ( m_origin != 0 )
      m_origin->memPool()->updateAlloc( nSize - m_gcSize );
   m_gcSize = nSize;
}

}


/* end of garbageable.cpp */
