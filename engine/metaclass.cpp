/*
   FALCON - The Falcon Programming Language.
   FILE: metaclass.cpp

   Handler for class instances.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 18 Jun 2011 21:24:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/metaclass.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/falconclass.h>
#include <falcon/falconinstance.h>


namespace Falcon {

MetaClass::MetaClass():
   Class("Class", FLC_CLASS_ID_CLASS )
{
}


MetaClass::~MetaClass()
{
}

void MetaClass::gcMark( void* self, uint32 mark ) const
{
   static_cast<Class*>(self)->gcMarkMyself( mark );
}

bool MetaClass::gcCheck( void* self, uint32 mark ) const
{
   return static_cast<Class*>(self)->gcCheckMyself( mark );
}

void MetaClass::dispose( void* self ) const
{
   delete static_cast<Class*>(self);
}


void* MetaClass::clone( void* source ) const
{
   return source;
}


void MetaClass::serialize( DataWriter*, void*  ) const
{
   // TODO
}


void* MetaClass::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}

void MetaClass::describe( void* instance, String& target, int, int ) const
{
   Class* fc = static_cast<Class*>(instance);
   target = "Class " + fc->name();
}

//====================================================================
// Operator overloads
//


void MetaClass::op_isTrue( VMContext* ctx, void* ) const
{
   // classes are always true
   ctx->topData().setBoolean(true);
}

void MetaClass::op_toString( VMContext* ctx , void* item ) const
{
   Class* fc = static_cast<Class*>(item);
   String* sret = new String( "Class " );
   sret->append(fc->name());
   ctx->topData() = sret;
}


void MetaClass::op_call( VMContext* ctx, int32 pcount, void* self ) const
{
   Class* fc = static_cast<Class*>(self);
   fc->op_create( ctx, pcount );
}

}

/* end of metaclass.cpp */
