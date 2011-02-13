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

namespace Falcon {

Class::Class( const String& name ):
   m_name( name ),
   m_typeID( FLC_CLASS_ID_OBJECT ),
   m_quasiFlat( false )
{}

Class::Class( const String& name, int64 tid ):
   m_name( name ),
   m_typeID( tid ),
   m_quasiFlat( false )
{}


Class::~Class()
{
}


void Class::gcMark( void* self, uint32 mark ) const
{
   // normally does nothing
}


void Class::describe( void* instance, String& target ) const
{
   target = "<*?>";
}

bool Class::getProperty( void* self, const String& property, Item& value ) const
{
   return false;
}

bool Class::setProperty( void* self, const String& property, const Item& value ) const
{
   return false;
}

bool Class::getIndex( void* self, const Item& index, Item& value ) const
{
   return false;
}

bool Class::setIndex( void* self, const Item& index, const Item& value ) const
{
   return false;
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


int Class::compare( void* self, const Item& value ) const
{
   Class* tgt;
   void* inst;
   switch( value.type() )
   {
      case FLC_ITEM_DEEP:
         if( (inst = value.asDeepInst()) == self )
         {
            return 0; // the same.
         }

         tgt = value.asDeepClass();
         break;
      case FLC_ITEM_USER:
         if( (inst = value.asUserInst()) == self )
         {
            return 0; // the same.
         }

         tgt = value.asUserClass(); break;
         break;
         
      default:
         // it's an ID < than us, that's for sure
         return 1;
   }

   // see the class signature.
   if ( typeID() != tgt->typeID() )
      return typeID() - tgt->typeID();

   // see the pointer
   if( self < inst ) return -1;
   // they can't be the same, we already checked it.
   return 1;
}


void* Class::assign( void* instance ) const
{
   // normally does nothing
   return instance;
}

//=====================================================================
// VM Operator override.
//

void Class::op_neg( VMachine *vm, void* self, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("neg") ) );
}

void Class::op_add( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("add") ) );
}

void Class::op_sub( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("sub") ) );
}


void Class::op_mul( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("mul") ) );
}


void Class::op_div( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("div") ) );
}


void Class::op_mod( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("mod") ) );
}


void Class::op_pow( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("pow") ) );
}


void Class::op_aadd( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("aadd") ) );
}


void Class::op_asub( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("asub") ) );
}


void Class::op_amul( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("amul") ) );
}


void Class::op_adiv( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("/=") ) );
}


void Class::op_amod( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("%=") ) );
}


void Class::op_apow( VMachine *vm, void* self, Item& op2, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("**=") ) );
}


void Class::op_inc(VMachine *vm, void* self, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("++x") ) );
}


void Class::op_dec(VMachine *vm, void* self, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("--x") ) );
}


void Class::op_incpost(VMachine *vm, void* self, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("x++") ) );
}


void Class::op_decpost(VMachine *vm, void* self, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("x--") ) );
}


void Class::op_call( VMachine *vm, int32 paramCount, void* self, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("()") ) );
}


void Class::op_getIndex(VMachine *vm, void* self, Item& idx, Item& target ) const
{
   if( ! getIndex(self, idx, target ) )
   {
      // TODO: IS it worth to add more infos about self in the error?
      vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("[]") ) );
   }
}


void Class::op_setIndex(VMachine *vm, void* self, Item& idx, Item& target ) const
{
   if( ! setIndex( self, idx, target ) )
   {
      // TODO: IS it worth to add more infos about self in the error?
      vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("[]=") ) );
   }
}


void Class::op_getProperty( VMachine *vm, void* self, const String& pname, Item& target ) const
{
   if( ! getProperty( self, pname, target ) )
   {
      // TODO: IS it worth to add more infos about self in the error?
      vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra(".") ) );
   }
}


void Class::op_setProperty( VMachine *vm, void* self, const String& pname, Item& target ) const
{
   if( ! setProperty( self, pname, target ) )
   {
      // TODO: IS it worth to add more infos about self in the error?
      vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra(".=") ) );
   }
}


void Class::op_lt( VMachine *vm, void* self, Item& op2, Item& target )const
{
   target.setBoolean( compare(self, op2) < 0 );
}

void Class::op_le( VMachine *vm, void* self, Item& op2, Item& target )const
{
   target.setBoolean( compare(self, op2) <= 0 );
}

void Class::op_gt( VMachine *vm, void* self, Item& op2, Item& target )const
{
   target.setBoolean( compare(self, op2) > 0 );
}

void Class::op_ge( VMachine *vm, void* self, Item& op2, Item& target )const
{
   target.setBoolean( compare(self, op2) >= 0 );
}

void Class::op_eq( VMachine *vm, void* self, Item& op2, Item& target )const
{
   target.setBoolean( compare(self, op2) == 0 );
}

void Class::op_ne( VMachine *vm, void* self, Item& op2, Item& target )const
{
   target.setBoolean( compare(self, op2) != 0 );
}

void Class::op_isTrue( VMachine *vm, void* self, Item& target ) const
{
   target.setBoolean(true);
}


void Class::op_in( VMachine *vm, void* self, Item& item, Item& target ) const
{
   vm->raiseError( new OperandError( ErrorParam(__LINE__, e_invop ).extra("in") ) );
}


void Class::op_provides( VMachine *vm, void* self, const String &pname, Item& target ) const
{
   target.setBoolean( hasProperty(self, pname) );
}

void Class::op_toString( VMachine *vm, void* self, Item& target ) const
{
   String *descr = new String();
   describe( self, *descr );
   target = descr->garbage();
}

}

/* end of class.cpp */
