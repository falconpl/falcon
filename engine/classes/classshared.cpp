/*
   FALCON - The Falcon Programming Language.
   FILE: classshared.cpp

   Interface for script to Shared variables.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 30 Nov 2012 12:52:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classes/classshared.h>
#include <falcon/shared.h>

namespace Falcon
{

ClassShared::ClassShared( const String& name ):
         ClassUser(name)
{
}

ClassShared::ClassShared( const String& name, int64 type ):
         ClassUser(name, type)
{
}


ClassShared::ClassShared():
         ClassUser("Shared")
{
}


ClassShared::~ClassShared()
{
}


void ClassShared::dispose( void* self ) const
{
   Shared* sh = static_cast<Shared*>(self);
   sh->decref();
}

void* ClassShared::clone( void* source ) const
{
   Shared* sh = static_cast<Shared*>(source);
   return sh->clone();
}


void* ClassShared::createInstance() const
{
   // this is a virtual class.
   return 0;
}

void ClassShared::describe( void* instance, String& target, int, int ) const
{
   target.A("<").A(name()).A("* ").N((int64) instance).A(">");
}

void ClassShared::gcMarkInstance( void* self, uint32 mark ) const
{
   Shared* sh = static_cast<Shared*>(self);
   sh->gcMark( mark );
}

bool ClassShared::gcCheckInstance( void* self, uint32 mark ) const
{
   Shared* sh = static_cast<Shared*>(self);
   return sh->gcMark() >= mark;
}


}

/* end of classshared.cpp */
