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

#include <falcon/datawriter.h>
#include <falcon/datareader.h>


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
 
void* ClassClosure::createInstance() const
{
   return new Closure;
}

void ClassClosure::store( VMContext*, DataWriter* wr, void* instance ) const
{
   Closure* cls = static_cast<Closure*>(instance);
   wr->write( cls->m_closedLocals );
}

void ClassClosure::restore( VMContext* ctx, DataReader* rd ) const
{
   Closure* cls = new Closure();
   uint32 locals;
   rd->read( locals );
   cls->m_closedLocals = locals;
   ctx->pushData( FALCON_GC_STORE( this, cls) );
}

void ClassClosure::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   Closure* closure = static_cast<Closure*>(instance);   
   uint32 size = closure->m_closedDataSize;
   // save also the current values of the items
   // TODO: If we save the ItemReference, the flattening mechanism should be able
   // to preserve the references in unflatten.
   subItems.resize( 1 + size );
   subItems[0] = Item( closure->m_handler, closure->m_closed );
   for(uint32 i = 0; i < size; ++i )
   {
      subItems[i+1] = *closure->closedData()[i].value();
   }
}


void ClassClosure::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{   
   Closure* closure = static_cast<Closure*>(instance);
   fassert( subItems.length() > 0 );
   
   subItems[0].forceClassInst( closure->m_handler, closure->m_closed );
   if( subItems.length() > 1 ) {
      closure->m_closedDataSize = subItems.length()-1;
      closure->m_closedData = new Variable[closure->m_closedDataSize];
      for(uint32 i = 1; i < subItems.length(); ++i )
      {  
         //TODO: Save as class Reference
         Variable v;
         v.value(&subItems[i]);
         closure->m_closedData[i-1].makeReference(&v);
      }      
   }
}
   
void ClassClosure::describe( void* instance, String& target, int depth, int maxlen) const
{
   Closure* closure = static_cast<Closure*>(instance);
   if( closure->m_handler == 0 || closure->m_closed == 0 )
   {
      target = "<Blank Closure>";
   }
   else {
      String temp;
      closure->m_handler->describe(closure->m_closed, temp, depth+1, maxlen );
      target = "/* Closure for */" + temp;
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
   ctx->call( closure, paramCount );
}

}

/* end of classclosure.cpp */
