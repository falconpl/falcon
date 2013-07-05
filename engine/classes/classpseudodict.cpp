/*
   FALCON - The Falcon Programming Language.
   FILE: classpseudodict.cpp

   Dictioanry-as-pseudo-object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classpseudodict.cpp"


#include <falcon/classes/classpseudodict.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/itemdict.h>
#include <falcon/stdhandlers.h>
#include <falcon/stderrors.h>

namespace Falcon {

static void get_dict( const Class*, const String&, void* instance, Item& value )
{
   static Class* cls = Engine::instance()->stdHandlers()->dictClass();
   value.setUser(cls,instance);
}


namespace _classDictionary {

/**
 @class PseudoDictionary
 @from Dictionary(...)

 This class acts as a dictionary, with the ability to access
 values in the entity through the '.' dot accessor, and
 creating a method in case the accessed entity is a function.
*/

}


ClassPseudoDict::ClassPseudoDict():
   ClassDict("PseudoDictionary")
{
   addProperty( "dict", &get_dict );
}


ClassPseudoDict::~ClassPseudoDict()
{
}


void ClassPseudoDict::restore( VMContext* ctx, DataReader* stream ) const
{
   ClassDict::restore( ctx, stream );
   ctx->topData().setUser( this, ctx->topData().asInst() );
}


void ClassPseudoDict::describe( void* instance, String& target, int maxDepth, int maxLen ) const
{
   if( maxDepth == 0 )
   {
      target = "...";
      return;
   }

   ItemDict* dict = static_cast<ItemDict*>(instance);
   dict->describe(target, maxDepth, maxLen);
}


bool ClassPseudoDict::hasProperty( void* instance, const String& prop ) const
{
   ItemDict* dict = static_cast<ItemDict*>(instance);
   return dict->find(Item(prop.handler(),const_cast<String*>(&prop))) != 0;
}

//=======================================================================
//


void ClassPseudoDict::op_setProperty( VMContext* ctx, void* instance, const String& prop) const
{
   ItemDict* dict = static_cast<ItemDict*>(instance);
   ctx->popData();  // remove self
   ConcurrencyGuard::Reader gr(ctx,dict->guard());
   dict->insert( FALCON_GC_HANDLE(new String(prop)), ctx->topData() );
   // leave the value on top
}


void ClassPseudoDict::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   ItemDict* dict = static_cast<ItemDict*>(instance);
   Item* find = dict->find( Item( prop.handler(), const_cast<String*>(&prop) ) );
   if( find == 0 ) {
      ClassDict::op_getProperty( ctx, instance, prop );
   }
   else {
      if( find->isFunction() )
      {
         ctx->topData().methodize( find->asFunction() );
      }
      else {
         ctx->topData() = *find;
      }
   }
}

void ClassPseudoDict::op_toString( VMContext* ctx, void* self ) const
{
   String s;
   s.A("[Pdict of ").N((int64)static_cast<ItemDict*>(self)->size()).A(" elements]");
   ctx->stackResult( 1, s );
}
}

/* end of classpseudodict.cpp */
