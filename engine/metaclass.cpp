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

void* MetaClass::createInstance() const
{
   return 0;
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
   static Collector* coll = Engine::instance()->collector();

   Class* fc = static_cast<Class*>(self);
   void* instance = fc->createInstance();
   if( instance == 0 )
   {
      if( fc->isFlatInstance() )
      {
         // ok, nothing to worry about. Pass the right data to the init.
         fc->op_init( ctx, ctx->opcodeParams( pcount + 1 ), pcount );
      }
      else {
         // non-instantiable class.
         throw new CodeError( ErrorParam(e_abstract_init, __LINE__, SRC )
            .extra( fc->name() )
            .origin( ErrorParam::e_orig_vm ) );
      }
   }
   else 
   {
      // save the deep instance and handle it to the collector
      Item* params = ctx->opcodeParams( pcount + 1 );
      params->setUser( fc, instance );
      FALCON_GC_STORE( coll, fc, instance );
      
      // finally, invoke init.
      if( fc->op_init( ctx, pcount ) )
      {
         // if init returned true, this means it went deep and will take care 
         // of the parameters.
         return;
      }
   }
   
   // if we're here, we can get rid of the parameters.
   ctx->popData( pcount );
}

}

/* end of metaclass.cpp */
