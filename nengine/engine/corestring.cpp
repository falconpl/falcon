/*
   FALCON - The Falcon Programming Language.
   FILE: corestring.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/corestring.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>

#include "falcon/optoken.h"

namespace Falcon {

CoreString::CoreString():
   Class("String", FLC_CLASS_ID_STRING )
{
}


CoreString::~CoreString()
{
}


void* CoreString::create(void* creationParams ) const
{
   cpars* cp = static_cast<cpars*>(creationParams);
   String* res = new String( cp->m_other );
   if ( cp->m_bufferize )
   {
      res->bufferize();
   }
   return res;
}


void CoreString::dispose( void* self ) const
{
   String* f = static_cast<String*>(self);
   delete f;
}


void* CoreString::clone( void* source ) const
{
   String* s = static_cast<String*>(source);
   return new String(*s);
}


void CoreString::serialize( DataWriter* stream, void* self ) const
{
   String* s = static_cast<String*>(self);
   s->serialize(stream);
}


void* CoreString::deserialize( DataReader* stream ) const
{
   String* s = new String;
   try {
      s->deserialize( stream );
   }
   catch( ... )
   {
      delete s;
      throw;
   }

   return s;
}

void CoreString::describe( void* instance, String& target, int, int maxlen ) const
{
   String* self = static_cast<String*>(instance);
   target.size(0);
   target.append('"');
   if( self->length() > maxlen )
   {
      target += self->subString(0, maxlen);
      target += "...";
   }
   else
   {
      target += *self;
   }
   
   target.append('"');
}

//=======================================================================
// Addition

void CoreString::op_add( VMachine *vm, void* self ) const
{
   String* str = static_cast<String*>(self);
   Item* op1, *op2;
   vm->operands(op1, op2);

   Class* cls;
   void* inst;
   if( ! op2->asClassInst( cls, inst ) )
   {
      String* copy = new String(*str);
      copy->append(op2->describe());
      vm->stackResult(2, copy->garbage() );
      return;
   }

   if ( cls->typeID() == typeID() )
   {
      // it's a string!
      String *copy = new String(*str);
      copy->append( *static_cast<String*>(inst) );
      vm->stackResult(2, copy->garbage() );
      return;
   }

   // else we surrender, and we let the virtual system to find a way.
   vm->ifDeep( &m_nextOp );
   // use a new slot to get the result
   VMContext* ctx = vm->currentContext();

   // this will transform op2 slot into its string representation.
   cls->op_toString( vm, inst );
   
   if( ! vm->wentDeep() )
   {
      // op2 has been transformed
      String* deep = (String*)(op2->type() == FLC_ITEM_DEEP ? op2->asDeepInst() : op2->asUserInst());
      deep->prepend( *str );
   }
}

//=======================================================================
// Auto Addition
//

void CoreString::op_aadd( VMachine *vm, void* self ) const
{
   String* str = static_cast<String*>(self);
   Item* op1, *op2;
   vm->operands(op1, op2);

   Class* cls;
   void* inst;
   if( ! op2->asClassInst( cls, inst ) )
   {
      if( op1->copied() )
      {
         String* copy = new String(*str);
         copy->append(op2->describe());
         vm->stackResult(2, copy->garbage() );
      }
      else
      {
         op1->asString()->append(op2->describe());
      }

      return;
   }

   if ( cls->typeID() == typeID() )
   {
      // it's a string!
      if( op1->copied() )
      {
         String *copy = new String(*str);
         copy->append( *static_cast<String*>(inst) );
         vm->stackResult(2, copy->garbage() );
      }
      else
      {
         op1->asString()->append( *static_cast<String*>(inst) );
      }
      return;
   }

   // else we surrender, and we let the virtual system to find a way.
   vm->ifDeep( &m_nextOp );
   // use a new slot to get the result
   VMContext* ctx = vm->currentContext();

   // this will transform op2 slot into its string representation.
   cls->op_toString( vm, inst );

   if( ! vm->wentDeep() )
   {
      // op2 has been transformed
      String* deep = (String*)(op2->type() == FLC_ITEM_DEEP ? op2->asDeepInst() : op2->asUserInst());
      deep->prepend( *str );
   }
}

CoreString::NextOp::NextOp()
{
   apply = apply_;
}


void CoreString::NextOp::apply_( const PStep*, VMachine* vm )
{
   // The result of a deep call is in A
   Item* op1, *op2;
   vm->operands(op1, op2); // we'll discard op2

   String* deep = vm->regA().asString();
   String* self = op1->asString();

   if( op1->copied() )
   {
      String* copy = new String(*self);
      copy->append( *deep );
      vm->stackResult( 2, copy->garbage() );
   }
   else
   {
      vm->currentContext()->popData();
      self->append(*deep);
   }
}

//=======================================================================
// Comparation
//

void CoreString::op_compare( VMachine *vm, void* self ) const
{
   Item* op1, *op2;
   OpToken token( vm, op1, op2 );
   String* string = static_cast<String*>(self);

   Class* otherClass;
   void* otherData;

   if( op2->asClassInst( otherClass, otherData ) )
   {
      if( otherClass->typeID() == typeID() )
      {
         token.exit( string->compare(*static_cast<String*>(otherData) ) );
      }
      else
      {
         token.exit( typeID() - otherClass->typeID() );
      }
   }
   else
   {
      token.exit( typeID() - op2->type() );
   }
}


void CoreString::op_toString( VMachine *vm, void* self ) const
{
   // nothing to do -- the topmost item of the stack is already a string.
}


}

/* end of corestring.cpp */
