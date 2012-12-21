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
#undef SRC
#define SRC "engine/classes/metaclass.cpp"

#include <falcon/classes/metaclass.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/falconclass.h>
#include <falcon/falconinstance.h>
#include <falcon/errors/codeerror.h>


namespace Falcon {

MetaClass::MetaClass():
   ClassMantra("$MetaClass", FLC_CLASS_ID_CLASS )
{
}


MetaClass::~MetaClass()
{
}


void MetaClass::describe( void* instance, String& target, int, int ) const
{
   Class* fc = static_cast<Class*>(instance);
   target = "Class " + fc->name();
}


Class* MetaClass::getParent( const String& name ) const
{
   Class* cls = Engine::instance()->mantraClass();
   
   if( name == cls->name() ) return cls;
   return 0;
}

bool MetaClass::isDerivedFrom( const Class* parent ) const
{
   Class* cls = Engine::instance()->mantraClass();
   
   return parent == cls || parent == this;
}

void MetaClass::enumerateParents( ClassEnumerator& cb ) const
{
   Class* cls = Engine::instance()->mantraClass();
   
   cb( cls, true );
}

void* MetaClass::getParentData( Class* parent, void* data ) const
{
   Class* cls = Engine::instance()->mantraClass();
   
   if( parent == cls || parent == this ) return data;
   return 0;
}

//====================================================================
// Operator overloads
//


void MetaClass::op_toString( VMContext* ctx , void* item ) const
{
   Class* fc = static_cast<Class*>(item);
   String* sret = new String( "Class " );
   sret->append(fc->name());
   ctx->topData().setUser( FALCON_GC_STORE( sret->handler(), sret ) );
}


void MetaClass::op_call( VMContext* ctx, int32 pcount, void* self ) const
{
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
      params->setUser( FALCON_GC_STORE( fc, instance ) );
      
      // finally, invoke init.
      if( fc->op_init( ctx, instance, pcount ) )
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
