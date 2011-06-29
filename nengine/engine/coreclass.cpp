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


void* CoreClass::create( void* ) const
{
   fassert2( false, "Cannot create a core class from creation parameters." );
   return 0;
}


void CoreClass::dispose( void* self ) const
{
   delete static_cast<FalconClass*>(self);
}


void* CoreClass::clone( void* source ) const
{
   return new FalconClass( *static_cast<FalconClass*>(source) );
}


void CoreClass::serialize( DataWriter* stream, void* self ) const
{
   static_cast<FalconClass*>(self)->serialize(stream);
}


void* CoreClass::deserialize( DataReader* stream ) const
{
   // TODO
   FalconClass* fi = new FalconClass;
   try
   {
      fi->deserialize(stream);
   }
   catch( ... )
   {
      delete fi;
      throw;
   }
   return fi;
}

void CoreClass::describe( void* instance, String& target, int, int ) const
{
   FalconClass* fc = static_cast<FalconClass*>(instance);
   target = "Class " + fc->name();
}

//====================================================================
// Operator overloads
void CoreClass::op_isTrue( VMachine *vm, void* ) const
{
   // classes are always true
   vm->currentContext()->topData().setBoolean(true);
}

void CoreClass::op_toString( VMachine *vm , void* item ) const
{
   FalconClass* fc = static_cast<FalconClass*>(item);
   String* sret = new String( "Class " );
   sret->append(fc->name());
   vm->currentContext()->topData() = sret;
}


void CoreClass::op_call( VMachine *vm, int32 /* pcount */, void* self ) const
{
   static Collector* coll = Engine::instance()->collector();
   static Class* clsInst = Engine::instance()->instanceClass();

   FalconClass* fc = static_cast<FalconClass*>(self);
   FalconInstance* fi = fc->createInstance();
   // save the intance so that it's already garbage-marked
   vm->currentContext()->topData().setDeep( coll->store(clsInst, fi) );
}

}

/* end of coreclass.cpp */
