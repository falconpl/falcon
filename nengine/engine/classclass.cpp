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

#include <falcon/classclass.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/falconclass.h>
#include <falcon/falconinstance.h>


namespace Falcon {

ClassClass::ClassClass():
   Class("Class", FLC_CLASS_ID_CLASS )
{
}


ClassClass::~ClassClass()
{
}


void ClassClass::dispose( void* self ) const
{
   delete static_cast<Class*>(self);
}


void* ClassClass::clone( void* source ) const
{
   return source;
}


void ClassClass::serialize( DataWriter*, void*  ) const
{
   // TODO
}


void* ClassClass::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}

void ClassClass::describe( void* instance, String& target, int, int ) const
{
   Class* fc = static_cast<Class*>(instance);
   target = "Class " + fc->name();
}

//====================================================================
// Operator overloads
//


void ClassClass::op_isTrue( VMContext* ctx, void* ) const
{
   // classes are always true
   ctx->topData().setBoolean(true);
}

void ClassClass::op_toString( VMContext* ctx , void* item ) const
{
   Class* fc = static_cast<Class*>(item);
   String* sret = new String( "Class " );
   sret->append(fc->name());
   ctx->topData() = sret;
}


void ClassClass::op_call( VMContext* ctx, int32 pcount, void* self ) const
{
   Class* fc = static_cast<Class*>(self);
   fc->op_create( ctx, pcount );
}

}

/* end of coreclass.cpp */
