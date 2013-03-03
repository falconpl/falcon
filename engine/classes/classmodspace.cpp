/*
   FALCON - The Falcon Programming Language.
   FILE: classmodspace.cpp

   Handler for dynamically created module spaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 05 Feb 2013 18:07:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classmodspace.cpp"

#include <falcon/fassert.h>
#include <falcon/vmcontext.h>
#include <falcon/errors/paramerror.h>
#include <falcon/classes/classmodspace.h>

#include <falcon/modspace.h>

namespace Falcon {

ClassModSpace::ClassModSpace():
         Class("ModSpace")
{
}

ClassModSpace::~ClassModSpace()
{
}

void* ClassModSpace::createInstance() const
{
   return 0;
}

void ClassModSpace::dispose( void* instance ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   spc->decref();
}

void* ClassModSpace::clone( void* instance ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   spc->incref();
   return spc;
}

void ClassModSpace::gcMarkInstance( void* instance, uint32 mark ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   spc->gcMark(mark);
}

bool ClassModSpace::gcCheckInstance( void* instance, uint32 mark ) const
{
   ModSpace* spc = static_cast<ModSpace*>(instance);
   return spc->currentMark() >= mark;
}
   
}

/* end of classmodspace.cpp */
