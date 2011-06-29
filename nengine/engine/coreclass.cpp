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
#include <falcon/coreinstance.h>
#include <falcon/itemid.h>
#include <falcon/vm.h>
#include <falcon/optoken.h>
#include <falcon/falconclass.h>
#include <falcon/falconinstance.h>


namespace Falcon {

CoreClass::CoreClass():
   Class("Class", FLC_CLASS_ID_CLASS )
{
}


CoreClass::~CoreClass()
{
}





void CoreClass::dispose( void* self ) const
{
   delete static_cast<Class*>(self);
}


void* CoreClass::clone( void* source ) const
{
   return source;
}


void CoreClass::serialize( DataWriter*, void*  ) const
{
   // TODO
}


void* CoreClass::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}

void CoreClass::describe( void* instance, String& target, int, int ) const
{
   Class* fc = static_cast<Class*>(instance);
   target = "Class " + fc->name();
}

//====================================================================
// Operator overloads
//


void CoreClass::op_isTrue( VMachine *vm, void* ) const
{
   // classes are always true
   vm->currentContext()->topData().setBoolean(true);
}

void CoreClass::op_toString( VMachine *vm , void* item ) const
{
   Class* fc = static_cast<Class*>(item);
   String* sret = new String( "Class " );
   sret->append(fc->name());
   vm->currentContext()->topData() = sret;
}


void CoreClass::op_call( VMachine *vm, int32 pcount, void* self ) const
{
   Class* fc = static_cast<Class*>(self);
   fc->op_create( vm, pcount );
}

}

/* end of coreclass.cpp */
