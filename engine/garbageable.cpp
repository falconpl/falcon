/*
   FALCON - The Falcon Programming Language
   FILE: garbageable.cpp
   $Id: garbageable.cpp,v 1.5 2007/08/11 00:11:54 jonnymind Exp $

   Garbageable objects, declaration.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun gen 22 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
