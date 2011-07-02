/*
   FALCON - The Falcon Programming Language.
   FILE: classfunction.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/classfunction.h>
#include <falcon/synfunc.h>
#include "falcon/itemid.h"

namespace Falcon {

ClassFunction::ClassFunction():
   Class("Function", FLC_CLASS_ID_FUNCTION )
{
}


ClassFunction::~ClassFunction()
{
}

void ClassFunction::dispose( void* self ) const
{
   Function* f = static_cast<Function*>(self);
   delete f;
}


void* ClassFunction::clone( void* ) const
{
   //Function* f = static_cast<Function*>(self);
   //TODO
   return 0;
}


void ClassFunction::serialize( DataWriter*, void*  ) const
{
   // TODO
}


void* ClassFunction::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}

void ClassFunction::describe( void* instance, String& target, int, int ) const
{
   Function* func = static_cast<Function*>(instance);
   target = func->name() + "()";
}

}

/* end of classfunction.cpp */
