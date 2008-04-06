/*
   FALCON - The Falcon Programming Language
   FILE: userdata.cpp

   Embeddable falcon object user data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab giu 23 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
