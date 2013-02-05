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

#include <falcon/itemarray.h>

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
   ClassUser(name),
   FALCON_INIT_PROPERTY( code ),
   FALCON_INIT_PROPERTY( description ),
   FALCON_INIT_PROPERTY( extra ),

   FALCON_INIT_PROPERTY( mantra ),
   FALCON_INIT_PROPERTY( module ),
   FALCON_INIT_PROPERTY( path ),
   FALCON_INIT_PROPERTY( signature ),
   FALCON_INIT_PROPERTY( line ),
   FALCON_INIT_PROPERTY( chr ),

   FALCON_INIT_PROPERTY( heading ),
   FALCON_INIT_PROPERTY( trace ),
   FALCON_INIT_PROPERTY( errors ),
   FALCON_INIT_PROPERTY( raised ),
   FALCON_INIT_METHOD( take )

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

void* ClassError::createInstance() const
{
   // ... and we're abstract.
   return 0;
}



bool ClassError::isDerivedFrom( const Class* cls ) const 
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


bool ClassError::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   Error* error = static_cast<Error*>( instance );
   ErrorParam params;
   invokeParams( ctx, pcount, params );
   error->set( params );
   return false;
}

void ClassError::op_toString( VMContext* ctx, void* self ) const
{
   Error* err = static_cast<Error*>(self);
   String* str = new String;   
   err->describeTo(*str);
   ctx->topData() = FALCON_GC_HANDLE(str);
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

//========================================================
// Property accessors
//

FALCON_DEFINE_PROPERTY_GET(ClassError, code)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   value.setInteger( error->errorCode() );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, code)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   error->errorCode( value.forceInteger() );
}

FALCON_DEFINE_PROPERTY_GET(ClassError, line)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   value.setInteger( error->line() );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, line)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   error->line( value.forceInteger() );
}

FALCON_DEFINE_PROPERTY_GET(ClassError, chr)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   value.setInteger( error->chr() );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, chr)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   error->chr( value.forceInteger() );
}


FALCON_DEFINE_PROPERTY_GET(ClassError, description)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   String* str = new String( error->errorDescription() );
   str->bufferize();
   value = FALCON_GC_HANDLE( str );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, description)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   if( value.isString() )
   {
      error->errorDescription( *value.asString() );
   }
   else {
      error->errorDescription( value.describe() );
   }
}

FALCON_DEFINE_PROPERTY_GET(ClassError, extra)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   String* str = new String( error->extraDescription() );
   str->bufferize();
   value = FALCON_GC_HANDLE( str );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, extra)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   if( value.isString() )
   {
      error->extraDescription( *value.asString() );
   }
   else {
      error->extraDescription( value.describe() );
   }
}

FALCON_DEFINE_PROPERTY_GET(ClassError, mantra)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   String* str = new String( error->mantra() );
   str->bufferize();
   value = FALCON_GC_HANDLE( str );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, mantra)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   if( value.isString() )
   {
      error->mantra( *value.asString() );
   }
   else {
      error->mantra( value.describe() );
   }
}

FALCON_DEFINE_PROPERTY_GET(ClassError, module)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   String* str = new String( error->module() );
   str->bufferize();
   value = FALCON_GC_HANDLE( str );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, module)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   if( value.isString() )
   {
      error->module( *value.asString() );
   }
   else {
      error->module( value.describe() );
   }
}

FALCON_DEFINE_PROPERTY_GET(ClassError, path)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   String* str = new String( error->path() );
   str->bufferize();
   value = FALCON_GC_HANDLE( str );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, path)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   if( value.isString() )
   {
      error->path( *value.asString() );
   }
   else {
      error->path( value.describe() );
   }
}

FALCON_DEFINE_PROPERTY_GET(ClassError, signature)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   String* str = new String( error->signature() );
   str->bufferize();
   value = FALCON_GC_HANDLE( str );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, signature)( void* instance, const Item& value )
{
   Error* error = static_cast<Error*>( instance );
   if( value.isString() )
   {
      error->sign( *value.asString() );
   }
   else {
      error->sign( value.describe() );
   }
}

FALCON_DEFINE_PROPERTY_GET(ClassError, heading)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );
   String* str = new String;
   error->heading(*str);
   value = FALCON_GC_HANDLE( str );
}

FALCON_DEFINE_PROPERTY_SET(ClassError, heading)( void*, const Item& )
{
   throw readOnlyError();
}


FALCON_DEFINE_PROPERTY_GET(ClassError, trace)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );

   if( error->hasTraceback() )
   {
      ItemArray* ret = new ItemArray;
      class TB: public Error::StepEnumerator {
      public:
         TB( ItemArray* ret ): m_ret(ret)  {}
         virtual ~TB() {}
         virtual bool operator()( const TraceStep& data, bool  )
         {
            String* tgt = new String;
            data.toString(*tgt);
            m_ret->append( FALCON_GC_HANDLE( tgt ) );
            return true;
         }

      private:
         ItemArray* m_ret;
      };

      TB rator(ret);
      error->enumerateSteps(rator);

      value = FALCON_GC_HANDLE( ret );
   }
   else {
      value.setNil();
   }
}

FALCON_DEFINE_PROPERTY_SET(ClassError, trace)( void* , const Item&  )
{
   throw readOnlyError();
}

FALCON_DEFINE_PROPERTY_GET(ClassError, errors)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );

   if( error->hasSubErrors() )
   {
      ItemArray* ret = new ItemArray;
      class ERR: public Error::ErrorEnumerator {
      public:
         ERR( ItemArray* ret ): m_ret(ret)  {}
         virtual ~ERR() {}
         virtual bool operator()( const Error& error, bool  )
         {
            Error* data = const_cast<Error*>(&error);
            data->incref();
            m_ret->append( FALCON_GC_HANDLE( data ) );
            return true;
         }

      private:
         ItemArray* m_ret;
      };

      ERR rator(ret);
      error->enumerateErrors(rator);

      value = FALCON_GC_HANDLE( ret );
   }
   else {
      value.setNil();
   }
}

FALCON_DEFINE_PROPERTY_SET(ClassError, errors)( void* , const Item&  )
{
   throw readOnlyError();
}

FALCON_DEFINE_PROPERTY_GET(ClassError, raised)( void* instance, Item& value )
{
   Error* error = static_cast<Error*>( instance );

   if( error->hasRaised() )
   {
      value = error->raised();
   }
   else {
      value.setNil();
   }
}

FALCON_DEFINE_PROPERTY_SET(ClassError, raised)( void* , const Item&  )
{
   throw readOnlyError();
}


FALCON_DEFINE_METHOD_P1( ClassError, take )
{
   Error* error = static_cast<Error*>( ctx->self().asInst() );
   ctx->contextualize(error, true);
   ctx->addTrace(error);
}

}

/* end of classerror.cpp */

