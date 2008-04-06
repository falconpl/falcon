/*
   FALCON - The Falcon Programming Language.
   FILE: suserdata.cpp

   Embeddable falcon object user data - shared version
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 20 Mar 2008 21:20:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Embeddable falcon object user data.
*/

#include <falcon/suserdata.h>
#include <falcon/mempool.h>

namespace Falcon {

SharedUserData::SharedUserData( VMachine *vm ):
   Garbageable( vm )
{}

SharedUserData::~SharedUserData()
{}

bool SharedUserData::shared() const
{
   return true;
}

void SharedUserData::gcMark( MemPool *mp )
{
   mark( mp->currentMark() );
}

}

/* end of suserdata.cpp */
