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
   Class* cls = Engine::instance()->mantraClass();
   
   if( name == cls->name() ) return cls;
   return 0;
}

bool ClassFunction::isDerivedFrom( const Class* parent ) const
{
   Class* cls = Engine::instance()->mantraClass();
   
   return parent == cls || parent == this;
}

void ClassFunction::enumerateParents( ClassEnumerator& cb ) const
{
   Class* cls = Engine::instance()->mantraClass();
   
   cb( cls, true );
}

void* ClassFunction::getParentData( Class* parent, void* data ) const
{
   Class* cls = Engine::instance()->mantraClass();
   
   if( parent == cls || parent == this ) return data;
   return 0;
}


void ClassFunction::describe( void* instance, String& target, int, int ) const
{
   Function* func = static_cast<Function*>(instance);
   target = func->name() + "()";
}



void ClassFunction::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   ctx->call( static_cast<Function*>(self), paramCount );
}


void ClassFunction::op_eval( VMContext* ctx, void* self ) const
{
   // called object is on top of the stack 
   ctx->call( static_cast<Function*>(self), 0 );
}

}

/* end of classfunction.cpp */
