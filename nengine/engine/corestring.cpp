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


void CoreString::serialize( Stream* stream, void* self ) const
{
   String* s = static_cast<String*>(self);
   s->serialize(stream);
}


void* CoreString::deserialize( Stream* stream ) const
{
   String* s = new String;
   try {
      s->deserialize( stream, false );
   }
   catch( ... )
   {
      delete s;
      throw;
   }

   return s;
}

void CoreString::describe( void* instance, String& target ) const
{
 target = *static_cast<String*>(instance);
}

//=======================================================================
//

void CoreString::op_add( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   String* str = static_cast<String*>(self);

   Class* cls;
   void* inst;
   switch( op2.type() )
   {
      case FLC_ITEM_DEEP:
         cls = op2.asDeepClass();
         inst = op2.asDeepInst();
         break;
      case FLC_ITEM_USER:
         cls = op2.asUserClass();
         inst = op2.asUserInst();
         break;

      default:
      {
         String* copy = new String(*str);
         copy->append(op2.describe());
         target = copy->garbage();
      }
   }

   vm->ifDeep( &m_nextOp );
   cls->op_toString( vm, inst, target );
   if( ! vm->wentDeep() )
   {
       String* deep = (String*)(target.type() == FLC_ITEM_DEEP ? target.asDeepInst() : target.asUserInst());
       deep->prepend( *str );
       //target = copy.garbage();
   }
}


CoreString::NextOp::NextOp()
{
   apply = apply_;
}

void CoreString::NextOp::apply_( const PStep*, VMachine* vm )
{
   const Item& regA = vm->regA();
   Item& topItem = vm->currentContext()->topData();
   String* deep = (String*)(regA.type() == FLC_ITEM_DEEP ? regA.asDeepInst() : regA.asUserInst());
   String* self = (String*)(topItem.type() == FLC_ITEM_DEEP ? topItem.asDeepInst() : topItem.asUserInst());
   String* copy = new String(*self);
   copy->append( *deep );
   topItem = copy->garbage();
}

}

/* end of corestring.cpp */
