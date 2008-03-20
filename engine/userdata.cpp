/*
   FALCON - The Falcon Programming Language
   FILE: userdata.cpp

   Embeddable falcon object user data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 23 2007
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
   Embeddable falcon object user data.
*/

#include <falcon/userdata.h>

namespace Falcon {

UserData::~UserData()
{}

bool UserData::isReflective() const
{
   return false;
}

bool UserData::shared() const
{
   return false;
}

void UserData::getProperty( VMachine *vm, const String &propName, Item &prop )
{
}

void UserData::setProperty( VMachine *vm, const String &propName, Item &prop )
{
}

UserData * UserData::clone() const
{
   return 0;
}

bool UserData::isSequence() const
{
   return false;
}

void UserData::gcMark( MemPool *mp )
{
}

}


/* end of userdata.cpp */
