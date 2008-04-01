/*
   FALCON - The Falcon Programming Language.
   FILE: suserdata.cpp

   Embeddable falcon object user data - shared version
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 20 Mar 2008 21:20:52 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
