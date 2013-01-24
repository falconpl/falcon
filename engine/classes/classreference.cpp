/*
   FALCON - The Falcon Programming Language.
   FILE: classreference.cpp

   Reference to remote items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Jul 2011 10:53:54 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classreference.cpp"

#include <falcon/fassert.h>
#include <falcon/classes/classreference.h>
#include <falcon/classes/classrawmem.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/itemid.h>
#include <falcon/errors/paramerror.h>
#include <falcon/variable.h>
#include <falcon/engine.h>

#include <falcon/itemarray.h>

namespace Falcon
{

ClassReference::ClassReference():
   Class("Reference", FLC_CLASS_ID_REF)
{   
}


ClassReference::~ClassReference()
{  
}

Class* ClassReference::getParent( const String& ) const
{
   return 0;
}

void ClassReference::dispose( void* self ) const
{
   Variable* ref = static_cast<Variable*>(self);
   // TODO: pool references
   // Do not delete the base: it's differently accounted.
   delete ref;
}


void* ClassReference::clone( void* source ) const
{
   // TODO: pool references
   return 0
}

void* ClassReference::createInstance() const
{
   Variable* ref = new Variable();
   return ref;
}

void ClassReference::store( VMContext*, DataWriter*, void* ) const {}
void ClassReference::restore( VMContext* ctx, DataReader* ) const
{
   ctx->pushData( Item( this, new Variable ) );
   // but don't set the base; that will be restored later.
}

void ClassReference::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   static ClassRawMem* rm = Engine::instance()->rawMemClass();
   Variable* ref = static_cast<Variable*>(instance);

   // save the raw memory we use to set the variable,
   // so different variables pointing to the same memory will be restored properly
   subItems.append(Item(rm, ref->base()));
}

void ClassReference::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   Variable* ref = static_cast<Variable*>(instance);
   Variable::t_aligned_variable* aligned = static_cast<Variable::t_aligned_variable*>(subItems[0].asInst());
   ref->base(&aligned->count);
   ref->value(&aligned->value);
}

void ClassReference::gcMarkInstance( void* self, uint32 mark ) const
{
   Variable* ref = static_cast<Variable*>(self);
   ref->gcMark(mark);
}


bool ClassReference::gcCheckInstance( void* self, uint32 mark ) const
{
   Variable* ref = static_cast<Variable*>(self);
   return ref->gcMark() >= mark;
}


void ClassReference::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   cb("value", false);
}


void ClassReference::enumeratePV( void* self, PVEnumerator& cb ) const
{
   Variable* ref = static_cast<Variable*>(self);
   cb( "value", *ref->value() );
}


bool ClassReference::hasProperty( void*, const String& prop ) const
{
   return prop == "value";
}


void ClassReference::describe( void* self, String& target, int depth, int maxlen ) const
{
   Variable* ref = static_cast<Variable*>(self);
   Class* cls;
   void* data;
   ref->value()->forceClassInst( cls, data );

   String temp;
   cls->describe( data, temp, depth+1, maxlen );
   target = "Ref{" + temp + "}";
}

   
bool ClassReference::op_init( VMContext* ctx, void* self, int32 pcount ) const
{
   Variable* ref = static_cast<Variable*>(self);
   if( pcount > 0 ) {
      Item* top = ctx->opcodeParams(pcount);
      // if the incoming data is a reference, reference it.
      if( typeID() == top->type() ) {
         Variable* otherRef = static_cast<Variable*>(top->asInst());
         ref->makeReference(otherRef);
      }
      else {
         Variable::makeFreeVariable(*ref);
         ref->value()->assign(*top);
      }
   }

   ctx->stackResult(pcount, Item(this, ref));
   return false;
}


void ClassReference::op_getProperty( VMContext* ctx, void* self, const String& prop) const
{
   if( prop == "value" ) {
      Variable* ref = static_cast<Variable*>(self);
      ctx->topData() = *ref->value();
   }
   else
   {
      Class::op_getProperty(ctx, self, prop);
   }
}


void ClassReference::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   ctx->popData();

   if( prop == "value" ) {
      Variable* ref = static_cast<Variable*>(self);
      Item& top = ctx->topData();
      if( top.type() == typeID() )
      {
         // we got a reference. Great.
         Variable* ref2 = static_cast<Variable*>(top.asInst());
         ref->makeReference( ref2 );
      }
      else {
         ref->value()->assign(top);
      }
      // Dereference the output
      top = *ref->value();
   }
   else
   {
      Class::op_getProperty(ctx, self, prop);
   }
}


void ClassReference::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   Variable* ref = static_cast<Variable*>(self);
   ctx->popData(paramCount);
   ctx->topData() = *ref->value();
}

}

/* end of classreference.cpp */
