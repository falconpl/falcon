/*
   FALCON - The Falcon Programming Language.
   FILE: falcondata.cpp

   Falcon common object reflection architecture.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jun 2008 11:09:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon common object reflection architecture.
*/

#include <falcon/falcondata.h>
#include <falcon/destroyable.h>
#include <falcon/stream.h>

namespace Falcon {
Destroyable::~Destroyable()
{}

bool FalconData::serialize( Stream *stream, bool bLive ) const
{
   return false;
}

bool FalconData::deserialize( Stream *stream, bool bLive )
{
   return false;
}

}

/* end of falcondata.cpp */
