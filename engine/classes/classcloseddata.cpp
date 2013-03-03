/*
   FALCON - The Falcon Programming Language.
   FILE: classcloseddata.h

   Handler for closure entities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 16:13:09 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/classcloseddata.cpp"

#include <falcon/classes/classcloseddata.h>
#include <falcon/closeddata.h>
#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>


namespace Falcon {

ClassClosedData::ClassClosedData():
   Class("ClosedData")
{}


ClassClosedData::~ClassClosedData()
{}


void ClassClosedData::dispose( void* self ) const
{
   delete static_cast<ClosedData*>(self);
}

void* ClassClosedData::clone( void* source ) const
{
   return static_cast<ClosedData*>(source)->clone();
}

void* ClassClosedData::createInstance() const
{
   return 0;
}


void ClassClosedData::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   ClosedData* closure = static_cast<ClosedData*>(instance);
   closure->flatten(ctx, subItems);
}


void ClassClosedData::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   ClosedData* closure = static_cast<ClosedData*>(instance);
   closure->unflatten(ctx, subItems, 0);
}

void ClassClosedData::describe( void* instance, String& target, int, int ) const
{
   ClosedData* closure = static_cast<ClosedData*>(instance);
   target = "/* Closed ";
   target.N( closure->size() );
   target += " itm */";
}


void ClassClosedData::gcMarkInstance( void* instance, uint32 mark ) const
{
   ClosedData* closure = static_cast<ClosedData*>(instance);
   closure->gcMark( mark );
}


bool ClassClosedData::gcCheckInstance( void* instance, uint32 mark ) const
{
   ClosedData* closure = static_cast<ClosedData*>(instance);
   return closure->gcMark() >= mark;
}


}

/* end of classcloseddata.cpp */
