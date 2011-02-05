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

#include <falcon/codeerror.h>

#include "falcon/error_messages.h"

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



bool Class::derivedFrom( Class* other ) const
{
   // todo
}


bool Class::hasProperty( const String& prop ) const
{
   return false;
}


void* Class::assign( void* instance ) const
{
   // normally does nothing
   return instance;
}


void Class::neg( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("neg") );
}

void Class::add( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("add") );
}

void Class::sub( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("sub") );
}


void Class::mul( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("mul") );
}


void Class::div( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("div") );
}


void Class::mod( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("mod") );
}


void Class::pow( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("pow") );
}


void Class::aadd( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("aadd") );
}


void Class::asub( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("asub") );
}


void Class::amul( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("amul") );
}


void Class::adiv( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("adiv") );
}


void Class::amod( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("amod") );
}


void Class::apow( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("apow") );
}


void Class::inc(VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("++x") );
}


void Class::dec(VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("--x") );
}


void Class::incpost(VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("x++") );
}


void Class::decpost(VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("x--") );
}


void Class::call( VMachine *vm, int32 paramCount ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("call") );
}


bool Class::getIndex(VMachine *vm ) const
{
   //TODO
}


bool Class::setIndex(VMachine *vm ) const
{
   //TODO
}


bool Class::getProperty( VMachine *vm ) const
{
   //TODO
}


void Class::setProperty( VMachine *vm ) const
{
   //TODO
}


int Class::compare( VMachine *vm )const
{
   //TODO
   return 0;
}


bool Class::isTrue( VMachine *vm ) const
{
   return false;
}


bool Class::in( VMachine *vm ) const
{
   throw new CodeError( ErrorParam(__LINE__, e_invop ).extra("in") );
}


}

/* end of class.cpp */
