/*
   FALCON - The Falcon Programming Language.
   FILE: class.cpp

   Class definition of a Falcon Class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Jan 2011 15:01:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/class.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>
#include <falcon/operanderror.h>
#include <falcon/error_messages.h>

#include <falcon/bom.h>

namespace Falcon {

Class::Class( const String& name ):
   m_name( name ),
   m_typeID( FLC_CLASS_ID_OBJECT ),
   m_falconClass( false )
{}

Class::Class( const String& name, int64 tid ):
   m_name( name ),
   m_typeID( tid ),
   m_falconClass( false )
{}


Class::~Class()
{
}


void Class::gcMark( void* self, uint32 mark ) const
{
   // normally does nothing
}


void Class::describe( void*, String& target, int, int ) const
{
   target = "<*?>";
}


void Class::enumerateProperties( void* self, Class::PropertyEnumerator& ) const
{
   // normally does nothing
}


bool Class::derivedFrom( Class* other ) const
{
   // todo
   return false;
}


bool Class::hasProperty( void* self, const String& prop ) const
{
   return false;
}


void Class::op_compare( VMachine *vm, void* self ) const
{
   void* inst;
   Item *op1, *op2;
   
   vm->operands( op1, op2 );
   
   switch( op2->type() )
   {
      case FLC_ITEM_DEEP:
         if( (inst = op2->asDeepInst()) == self )
         {
            vm->stackResult(2, (int64)0 );
            return;
         }

         if( typeID() > 0 )
         {
            vm->stackResult(2, (int64)  typeID() - op2->asDeepClass()->typeID() );
            return;
         }
         break;

      case FLC_ITEM_USER:
         if( (inst = op2->asUserInst()) == self )
         {
            vm->stackResult(2, 0 );
            return;
         }

         if( typeID() > 0 )
         {
            vm->stackResult(2, (int64)  typeID() - op2->asDeepClass()->typeID() );
            return;
         }
         break;
   }

   // we have no information about what an item might be here, but we can
   // order the items by type
   vm->stackResult(2, (int64) op1->type() - op2->type() );
}


//=====================================================================
// VM Operator override.
//

void Class::op_neg( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("neg") );
}

void Class::op_add( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("add") );
}

void Class::op_sub( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("sub") );
}


void Class::op_mul( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("mul") );
}


void Class::op_div( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("div") );
}


void Class::op_mod( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("mod") );
}


void Class::op_pow( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("pow") );
}


void Class::op_aadd( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("aadd") );
}


void Class::op_asub( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("asub") );
}


void Class::op_amul( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("amul") );
}


void Class::op_adiv( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("/=") );
}


void Class::op_amod( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("%=") );
}


void Class::op_apow( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("**=") );
}


void Class::op_inc( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("++x") );
}


void Class::op_dec( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("--x") );
}


void Class::op_incpost( VMachine *, void*) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("x++") );
}


void Class::op_decpost( VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("x--") );
}


void Class::op_call( VMachine *, int32, void* self ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_non_callable ) );
}


void Class::op_getIndex(VMachine *, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("[]") );
}


void Class::op_setIndex(VMachine *, void* ) const
{
   // TODO: IS it worth to add more infos about self in the error?
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("[]=") );
}


void Class::op_getProperty( VMachine* vm, void* data, const String& property ) const
{
   static BOM* bom = Engine::instance()->getBom();

   // try to find a valid BOM propery.
   BOM::handler handler = bom->get( property );
   if ( handler != 0  )
   {
      handler( vm, this, data );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_prop_acc ).extra(property) );
   }
}


void Class::op_setProperty( VMachine *, void*, const String& ) const
{
   // TODO: IS it worth to add more infos about self in the error?
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(".=") );
}


void Class::op_isTrue( VMachine *vm, void* ) const
{
   vm->stackResult(1, true);
}


void Class::op_in( VMachine *, void*) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("in") );
}


void Class::op_provides( VMachine *vm, void*, const String& ) const
{
   vm->stackResult(1, false);
}

void Class::op_toString( VMachine *vm, void *self ) const
{
   String *descr = new String();
   describe( self, *descr );
   vm->stackResult(1, descr->garbage());
}

}

/* end of class.cpp */
