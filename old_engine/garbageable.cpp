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

GarbageableBase::GarbageableBase( const GarbageableBase &other ):
   m_gcStatus( other.m_gcStatus )
{}

GarbageableBase::~GarbageableBase()
{}

bool GarbageableBase::finalize()
{
   delete this;
   return true;
}

uint32 GarbageableBase::occupation()
{
   return 0;
}

//========================================================================

Garbageable::Garbageable()
{
   memPool->storeForGarbage(this);
}

Garbageable::Garbageable( const Garbageable &other ):
   GarbageableBase( other )
{
   memPool->storeForGarbage(this);
}

Garbageable::~Garbageable()
{}

void Garbageable::gcMark( uint32 mk )
{
   mark( mk );
}

}


/* end of garbageable.cpp */
