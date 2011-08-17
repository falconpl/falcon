/*
   FALCON - The Falcon Programming Language.
   FILE: corefunc.cpp

   Abstract live function object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 07 Jan 2009 14:54:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Abstract live function object.
*/

#include <falcon/corefunc.h>
#include <falcon/vm.h>
#include <falcon/eng_messages.h>
#include <falcon/itemarray.h>

namespace Falcon
{

CoreFunc::~CoreFunc()
{
   delete m_closure;
}

void CoreFunc::readyFrame( VMachine* vm, uint32 paramCount )
{
   vm->prepareFrame( this, paramCount );
}

void CoreFunc::gcMark( uint32 gen )
{
   if( mark() != gen )
   {
      mark( gen );
      liveModule()->gcMark( gen );
      // mark also closed items
      if ( closure() != 0 )
         closure()->gcMark( gen );
   }
}

}

/* end of corefunc.cpp */
