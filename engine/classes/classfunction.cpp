/*
   FALCON - The Falcon Programming Language.
   FILE: classfunction.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classsfunction.cpp"

#include <falcon/trace.h>
#include <falcon/classes/classfunction.h>
#include <falcon/synfunc.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/module.h>
#include <falcon/pseudofunc.h>
#include <falcon/collector.h>

#include <falcon/itemarray.h>

#include <falcon/errors/ioerror.h>

namespace Falcon {

ClassFunction::ClassFunction():
   ClassMantra("Function", FLC_CLASS_ID_FUNC )
{
}


ClassFunction::~ClassFunction()
{
}


Class* ClassFunction::getParent( const String& name ) const
{
   static Class* cls = Engine::instance()->mantraClass();
   
   if( name == cls->name() ) return cls;
   return 0;
}

bool ClassFunction::isDerivedFrom( const Class* parent ) const
{
   static Class* cls = Engine::instance()->mantraClass();
   
   return parent == cls || parent == this;
}

void ClassFunction::enumerateParents( ClassEnumerator& cb ) const
{
   static Class* cls = Engine::instance()->mantraClass();
   
   cb( cls, true );
}

void* ClassFunction::getParentData( Class* parent, void* data ) const
{
   static Class* cls = Engine::instance()->mantraClass();
   
   if( parent == cls || parent == this ) return data;
   return 0;
}


void ClassFunction::enumerateProperties( void* instance, Class::PropertyEnumerator& cb ) const
{
   static Class* cls = Engine::instance()->mantraClass();

   cb("params", false);
   cls->enumerateProperties(instance, cb);
}


void ClassFunction::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{
   static Class* cls = Engine::instance()->mantraClass();
   cls->enumeratePV(instance, cb);
}


bool ClassFunction::hasProperty( void* instance, const String& prop ) const
{
   static Class* cls = Engine::instance()->mantraClass();

   return
         prop == "params"
         || cls->hasProperty( instance, prop );
}


void ClassFunction::describe( void* instance, String& target, int, int ) const
{
   Function* func = static_cast<Function*>(instance);
   target = func->name() + " /* Function " + func->locate() + " */";
}


void ClassFunction::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   static Class* cls = Engine::instance()->mantraClass();
   Function* func = static_cast<Function*>(instance);

   if( prop == "params" )
   {
      // todo: cache them?
      ItemArray* params = new ItemArray;
      const VarMap& vars = func->variables();
      for( uint32 i = 0; i < vars.paramCount(); ++i ) {
         params->append( FALCON_GC_HANDLE(new String(vars.getParamName(i) ) ) );
      }

      ctx->stackResult(1, FALCON_GC_HANDLE(params) );
   }
   else {
      cls->op_getProperty( ctx, instance, prop );
   }
}


void ClassFunction::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   ctx->callInternal( static_cast<Function*>(self), paramCount );
}

void ClassFunction::op_toString( VMContext* ctx, void* self ) const
{
   Function* func = static_cast<Function*>(self);
   String* ret = new String(func->name());
   ret->append("()");
   ctx->pushData( FALCON_GC_HANDLE( ret ) );
}

}

/* end of classfunction.cpp */
