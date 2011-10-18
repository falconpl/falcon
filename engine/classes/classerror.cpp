/*
   FALCON - The Falcon Programming Language.
   FILE: classerror.cpp

   Class for storing error in scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classerror.cpp"

#include <falcon/classes/classerror.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/errors/paramerror.h>
#include <falcon/function.h>
#include <falcon/module.h>

#include <falcon/stderrors.h>

namespace Falcon {

/** Base handler for script level error.
 
 Falcon extension code can throw C++ exceptions that might be intercepted by
 scripts; similary, scripts can raise Falcon items that can eventually be
 interpreted as C++ exceptions, and then rethrown at C++ level.
 
 This class is the Class handler for C++ exceptions known and thrown by the
 engine and extension modules.
 
 Each specific error that might be thrown by modules should have its own handler,
 which can be exposed to scripts so that scripts are also able to intercept
 and create the same errors that C++ can throw.
 
 The error Class handler for the exceptions known by the engine are all stored
 in the errorclasses.h header, and a global instance for them can be accessed
 in the StdErrors class collection (which is exposed by the Engine::stdErrors() 
 method).
 
 */
ClassError::ClassError( const String& name ):
   Class(name)
{
   m_bIsErrorClass = true;
}

ClassError::~ClassError()
{
}

void ClassError::dispose( void* self ) const
{
   // use the virtual delete feature.
   Error* error = (Error*)self;
   error->decref();
}

void* ClassError::clone( void* ) const
{
   // errors are uncloneable for now
   return 0;
}

void ClassError::serialize( DataWriter*, void* ) const
{
   // TODO
}

void* ClassError::deserialize( DataReader* ) const
{
   //TODO
   return 0;
}


bool ClassError::isDerivedFrom( Class* cls ) const 
{
   static Class* stdError = Engine::instance()->stdErrors()->error();   
   return cls == this || cls == stdError;
}


void ClassError::describe( void* instance, String& target, int, int maxlen ) const
{
   Error* err = static_cast<Error*>(instance);
   target.size(0);
   err->heading(target);
   target = "inst. of " + target;
   if( maxlen > 3 && target.length() > (uint32) maxlen )
   {
      target = target.subString( 0,  maxlen - 3 ) + "...";
   }
}


void ClassError::op_toString( VMContext* ctx, void* self ) const
{
   Error* err = static_cast<Error*>(self);
   String* str = new String;   
   err->describeTo(*str);
   ctx->topData() = str->garbage();
}


bool ClassError::invokeParams( VMContext* ctx, int pcount, ErrorParam& params, bool bThrow ) const
{
   Item IerrId;
   Item Idesc;
   Item Iextra;
   
   if( pcount == 1 )
   {
      IerrId = ctx->opcodeParam(0);
   }
   else if ( pcount == 2 )
   {
      IerrId = ctx->opcodeParam(1);
      Idesc = ctx->opcodeParam(2);
   }
   else if( pcount >= 3 )
   {
      IerrId = ctx->opcodeParam(pcount-1);
      Idesc = ctx->opcodeParam(pcount-2);
      Iextra = ctx->opcodeParam(pcount-3);
   }
   
   bool bError = false;
   int nErrId = 0;
   if( IerrId.isOrdinal() )
   {
      nErrId = IerrId.forceInteger();
   }
   else if( ! IerrId.isNil() )
   {
      bError = true;
   }
   
   String desc;
   if( Idesc.isString() )
   {
      desc = *Idesc.asString();
   }
   else if( ! Idesc.isNil() )
   {
      bError = true;
   }
   
   String extra;
   if( Iextra.isString() )
   {
      extra = *Iextra.asString();
   }
   else if( ! Iextra.isNil() )
   {
      bError = true;
   }
   
   if( bError )
   {
      if( bThrow )
      {
         throw new ParamError( 
            ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_vm ));
      }
      else
      {
         return false;
      }
   }
   
   // Ok, the parameters are standing.
   params.code( nErrId );
   params.desc( desc );
   params.extra( extra );
   
   // Usually, when created this way we're created by a script
   params.origin( ErrorParam::e_orig_script );
   
   // now get the current line, symbol and module from the stack.
   int line = 0;
   if( ctx->codeDepth() != 0 )
   {
      line = ctx->currentCode().m_step->line();
   }
   
   // and the current module/symbol pair
   if( ctx->callDepth() > 0 )
   {
      CallFrame& cf = ctx->currentFrame();
      if ( cf.m_function != 0 )
      {
         params.symbol( cf.m_function->name() );
         if( cf.m_function->module() != 0 )
         {
            params.module(cf.m_function->module()->name());
         }
         
         // If we couldn't get the line of the pstep, get the line of the function
         if( line == 0 )
         {
            line = cf.m_function->declaredAt();
         }
      }
   }
   
   params.line( line );
   
   return true;
}

}

/* end of classerror.cpp */

