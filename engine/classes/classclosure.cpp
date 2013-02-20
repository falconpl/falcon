/*
   FALCON - The Falcon Programming Language.
   FILE: classclosure.h

   Handler for closure entities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 16:13:09 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/closureclass.cpp"

#include <falcon/classes/classclosure.h>
#include <falcon/closure.h>
#include <falcon/function.h>
#include <falcon/itemarray.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>

#include <falcon/datawriter.h>
#include <falcon/datareader.h>


namespace Falcon {

ClassClosure::ClassClosure():
   Class("Closure", FLC_CLASS_ID_CLOSURE)
{}


ClassClosure::~ClassClosure()
{}
  

void ClassClosure::dispose( void* self ) const
{
   delete static_cast<Closure*>(self);
}

void* ClassClosure::clone( void* source ) const
{
   return static_cast<Closure*>(source)->clone();
}
 
void* ClassClosure::createInstance() const
{
   return new Closure;
}


void ClassClosure::flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   closure->flatten(ctx, subItems);
}


void ClassClosure::unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const
{   
   Closure* closure = static_cast<Closure*>(instance);
   closure->unflatten(ctx, subItems, 0);
}
   
void ClassClosure::describe( void* instance, String& target, int depth, int maxlen) const
{
   Closure* closure = static_cast<Closure*>(instance);
   target = "/* Closure ";
   if( closure->data() != 0 ) {
      target.N( closure->data()->size() );
      target += " itm ";
   }
   else {
      target += "empty";
   }

   Function* func = closure->closed();
   if( func != 0 ) {
      target += "for */ ";
      String temp;
      // don't descend in depth
      func->handler()->describe( func, temp, depth, maxlen );
      target += temp;
   }
   else {
      target +=" */";
   }
}


void ClassClosure::gcMarkInstance( void* instance, uint32 mark ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   closure->gcMark( mark );
}


bool ClassClosure::gcCheckInstance( void* instance, uint32 mark ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   return closure->gcMark() >= mark;
}

   
void ClassClosure::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   ctx->callInternal( closure, paramCount );
}

}

/* end of classclosure.cpp */
