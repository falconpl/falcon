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
#include <falcon/vmcontext.h>
#include <falcon/operanderror.h>
#include <falcon/error_messages.h>

#include <falcon/bom.h>

namespace Falcon {

Class::Class( const String& name ):
   m_name( name ),
   m_typeID( FLC_CLASS_ID_OBJECT ),
   m_module(0),
   m_lastGCMark(0)
{}

Class::Class( const String& name, int64 tid ):
   m_name( name ),
   m_typeID( tid ),
   m_module(0),
   m_lastGCMark(0)
{}


Class::~Class()
{
}


void Class::gcMark( void*, uint32 ) const
{
   // normally does nothing
}


bool Class::gcCheck( void*, uint32 ) const
{
   return true;
}


void Class::gcMarkMyself( uint32 mark )
{
   m_lastGCMark = mark;
}


bool Class::gcCheckMyself( uint32 mark )
{
   if( mark > m_lastGCMark )
   {
      delete this;
      return false;
   }

   return true;
}


void Class::describe( void*, String& target, int, int ) const
{
   target = "<*?>";
}


void Class::enumerateProperties( void*, Class::PropertyEnumerator& ) const
{
   // normally does nothing
}


bool Class::hasProperty( void*, const String& ) const
{
   return false;
}


void Class::op_compare( VMContext* ctx, void* self ) const
{
   void* inst;
   Item *op1, *op2;
   
   ctx->operands( op1, op2 );
   
   if( op2->isUser() )
   {
      if( (inst = op2->asInst()) == self )
      {
         ctx->stackResult(2, 0 );
         return;
      }

      if( typeID() > 0 )
      {
         ctx->stackResult(2, (int64)  typeID() - op2->asClass()->typeID() );
         return;
      }
   }

   // we have no information about what an item might be here, but we can
   // order the items by type
   ctx->stackResult(2, (int64) op1->type() - op2->type() );
}

//=====================================================================
// VM Operator override.
//

void Class::op_create( VMContext* , int32 ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("create") );
}


void Class::op_neg( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("neg") );
}

void Class::op_add( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("add") );
}

void Class::op_sub( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("sub") );
}


void Class::op_mul( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("mul") );
}


void Class::op_div( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("div") );
}


void Class::op_mod( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("mod") );
}


void Class::op_pow( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("pow") );
}


void Class::op_aadd( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("aadd") );
}


void Class::op_asub( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("asub") );
}


void Class::op_amul( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("amul") );
}


void Class::op_adiv( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("/=") );
}


void Class::op_amod( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("%=") );
}


void Class::op_apow( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("**=") );
}


void Class::op_inc( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("++x") );
}


void Class::op_dec( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("--x") );
}


void Class::op_incpost( VMContext* , void*) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("x++") );
}


void Class::op_decpost( VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("x--") );
}


void Class::op_call( VMContext* , int32, void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_non_callable ) );
}


void Class::op_getIndex(VMContext* , void* ) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("[]") );
}


void Class::op_setIndex(VMContext* , void* ) const
{
   // TODO: IS it worth to add more infos about self in the error?
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("[]=") );
}


void Class::op_getProperty( VMContext* ctx, void* data, const String& property ) const
{
   static BOM* bom = Engine::instance()->getBom();

   // try to find a valid BOM propery.
   BOM::handler handler = bom->get( property );
   if ( handler != 0  )
   {
      handler( ctx, this, data );
   }
   else
   {
      throw new OperandError( ErrorParam(__LINE__, e_prop_acc ).extra(property) );
   }
}


void Class::op_setProperty( VMContext* , void*, const String& ) const
{
   // TODO: IS it worth to add more infos about self in the error?
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra(".=") );
}


void Class::op_isTrue( VMContext* ctx, void* ) const
{
   ctx->stackResult(1, true);
}


void Class::op_in( VMContext* , void*) const
{
   throw new OperandError( ErrorParam(__LINE__, e_invop ).extra("in") );
}


void Class::op_provides( VMContext* ctx, void*, const String& ) const
{
   ctx->stackResult(1, false);
}

void Class::op_toString( VMContext* ctx, void *self ) const
{
   String *descr = new String();
   describe( self, *descr );
   ctx->stackResult(1, descr->garbage());
}

}

/* end of class.cpp */
