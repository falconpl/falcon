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
#include <falcon/vmcontext.h>



namespace Falcon {

ClassClosure::ClassClosure():
   Class("Closure")
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
   
void ClassClosure::store( VMContext*, DataWriter*, void* ) const
{}

void ClassClosure::restore( VMContext*, DataReader*, void*& empty ) const
{
   //TODO
   empty = new Closure();
}

void ClassClosure::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   Function* func = closure->function();
   
   SymbolTable& symtab = func->symbols();
   uint32 size = symtab.closedCount();
   // save also the current values of the items
   // TODO: If we save the ItemReference, the flattening mechanism should be able
   // to preserve the references in unflatten.
   subItems.resize( 1 + size );
   subItems[0].setFunction( func );
   for(uint32 i = 0; i < size; ++i )
   {
      subItems[i+1] = closure->closedData()[i];
   }
}


void ClassClosure::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{   
   Closure* closure = static_cast<Closure*>(instance);
   fassert( subItems.length() > 0 );
   fassert( subItems[0].isFunction() );
   
   closure->function( subItems[0].asFunction() );  
   for(uint32 i = 1; i < subItems.length(); ++i )
   {      
      closure->closedData()[i-1] = subItems[i];
   }
}
   
void ClassClosure::describe( void* instance, String& target, int, int) const
{
   Closure* closure = static_cast<Closure*>(instance);
   if( closure->function() == 0 )
   {
      target = "<Blank Closure>";
   }
   else {
      target = "<Closure for " + closure->function()->name() + ">";
   }
}


void ClassClosure::gcMark( void* instance, uint32 mark ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   closure->gcMark( mark );
}


bool ClassClosure::gcCheck( void* instance, uint32 mark ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   return closure->gcMark() >= mark;
}

   
void ClassClosure::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   Closure* closure = static_cast<Closure*>(instance);
   ctx->call( closure->function(), &closure->closedData(), paramCount );
}

}

/* end of classclosure.cpp */
