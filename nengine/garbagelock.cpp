/*
   FALCON - The Falcon Programming Language.
   FILE: garbagelock.cpp

   Garbage lock - safeguards for items in VMs.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 13 Jan 2011 00:26:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/garbagelock.h>

namespace Falcon {

GarbageLock::GarbageLock( bool )
{
}

GarbageLock::GarbageLock()
{
   //memPool->addGarbageLock( this );
}

GarbageLock::GarbageLock( const Item &itm ):
   m_item(itm)
{
   //memPool->addGarbageLock( this );
}

GarbageLock::~GarbageLock()
{
   //memPool->removeGarbageLock( this );
}

}

/* end of garbagelock.cpp */
