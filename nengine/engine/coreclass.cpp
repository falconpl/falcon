/*
   FALCON - The Falcon Programming Language.
   FILE: coreclass.cpp

   Class type handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 18 Jun 2011 21:24:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/coreclass.h>
#include <falcon/coreclass.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>
#include <falcon/optoken.h>

namespace Falcon {

CoreClass::CoreClass():
   Class("Class", FLC_CLASS_ID_CLASS )
{
}


CoreClass::~CoreClass()
{
}


void* CoreClass::create( void* creationParams ) const
{
}


void CoreClass::dispose( void* self ) const
{
}


void* CoreClass::clone( void* source ) const
{
}


void CoreClass::serialize( DataWriter* stream, void* self ) const
{
}


void* CoreClass::deserialize( DataReader* stream ) const
{
}

void CoreClass::describe( void* instance, String& target, int, int maxlen ) const
{
}

//====================================================================
// Operator overloads
void CoreString::op_true( VMachine *vm, void* ) const
{
   // classes are always true
   vm->currentContext()->topData()->setBoolean(true);
}

void CoreString::op_toString( VMachine *vm , void* item ) const
{
   
}


}

/* end of coreclass.cpp */
