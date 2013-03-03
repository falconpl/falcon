/*
   FALCON - The Falcon Programming Language.
   FILE: classrequirement.cpp

   Requirement object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 22 Feb 2012 19:50:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#undef SRC
#define SRC "engine/classes/classrequirement.cpp"

#include <falcon/classes/classrequirement.h>

#include "falcon/requirement.h"

namespace Falcon
{

ClassRequirement::ClassRequirement( const String& name ):
   Class(name)
{
}

ClassRequirement::~ClassRequirement()
{
}


void ClassRequirement::dispose( void* ) const
{
   // do nothing
}

void* ClassRequirement::clone( void* ) const
{
   // do nothing
   return 0;
}

void* ClassRequirement::createInstance() const
{
   // do nothing
   return 0;
}


void ClassRequirement::store( VMContext*, DataWriter* stream, void* instance ) const
{
   Requirement* req = static_cast<Requirement*>(instance);
   req->store(stream);
}


void ClassRequirement::restore( VMContext*, DataReader* stream, void*& empty ) const
{
   Requirement* req = static_cast<Requirement*>(empty);
   req->restore(stream);
}


void ClassRequirement::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   Requirement* req = static_cast<Requirement*>(instance);
   req->flatten(subItems);
}


void ClassRequirement::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   Requirement* req = static_cast<Requirement*>(instance);
   req->unflatten(subItems);
}


void ClassRequirement::describe( void* instance, String& target, int, int ) const
{
   Requirement* req = static_cast<Requirement*>(instance);
   target = "Requirement for \"" + req->name() + "\"";
}

}

/* end of classrequirement.cpp */
