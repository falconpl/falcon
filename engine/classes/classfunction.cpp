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
#include <falcon/stdhandlers.h>
#include <falcon/processor.h>
#include <falcon/classes/classmodule.h>
#include <falcon/module.h>
#include <falcon/modspace.h>
#include <falcon/textwriter.h>
#include <falcon/stringstream.h>
#include <falcon/stdsteps.h>


#include <falcon/itemarray.h>
#include <falcon/itemdict.h>

#include <falcon/stderrors.h>

namespace Falcon {

/*#
@class Function
@brief Reflects a function, possibly the current one.
@prop current  (static) Returns the current function (equivalent to fself)
@prop pcount (static) Count of parameters currently being passed to this function.
@prop params (static) Array of actual parameter values sent to the current function
@prop pdict (static) Dictionary of actual parameter values sent to the current function.
@prop vcount (static) Count of variable parameters (total parameters minus declared parameters).
@prop vp (static) Array of extra variable parameters sent to the current function.
@prop caller (static) The caller function or method of the current function (nil if none).
@prop invoker (static) Object from where the current function was invoked (nil if not an object).

@prop name Name of the given function
@prop fullname Name of the given function, including method prefixes.
@prop module Module where this function is declared (can be nil)
@prop location Standardized description of source code location where this function is declared.
@prop methodOf Parent class of this function, if it's a statically defined class method.
@prop plist Array containing the names of the parameters explicitly declared by this function.
@prop signature Explicitly declared prototype for this function.

@see passvp
*/


static void get_current( const Class*, const String&, void*, Item& value )
{
   VMContext* ctx = Processor::currentProcessor()->currentContext();
   Function* func = ctx->currentFrame().m_function;
   value.setUser(func->handler(), func);
}


static void get_pcount( const Class*, const String&, void*, Item& value )
{
   VMContext* ctx = Processor::currentProcessor()->currentContext();
   value = (int64)ctx->currentFrame().m_paramCount;
}

static void get_vcount( const Class*, const String&, void*, Item& value )
{
   VMContext* ctx = Processor::currentProcessor()->currentContext();
   CallFrame& frame = ctx->currentFrame();
   uint32 pdef = frame.m_function->paramCount();
   int64 pc = (int64) ctx->currentFrame().m_paramCount;
   pc -= pdef;
   if( pc < 0 ){ pc = 0;}
   value = pc;
}


static void get_params( const Class*, const String&, void*, Item& value )
{
   VMContext* ctx = Processor::currentProcessor()->currentContext();
   CallFrame& frame = ctx->currentFrame();
   ItemArray* array = new ItemArray(frame.m_paramCount);
   for( uint32 i = 0; i < frame.m_paramCount; ++i )
   {
      array->append(*ctx->param(i));
   }

   value = FALCON_GC_HANDLE(array);
}

static void get_vp( const Class*, const String&, void*, Item& value )
{
   VMContext* ctx = Processor::currentProcessor()->currentContext();
   CallFrame& frame = ctx->currentFrame();
   uint32 pdef = frame.m_function->paramCount();
   uint32 pcount = (uint32) frame.m_paramCount;

   ItemArray* array;
   if( pdef >= pcount ) {
      array = new ItemArray;
   }
   else {
      array = new ItemArray(pcount - pdef);
      for( uint32 i = pdef; i < pcount; ++i )
      {
         array->append(*ctx->param(i));
      }
   }

   value = FALCON_GC_HANDLE(array);
}



static void get_pdict( const Class*, const String&, void*, Item& value )
{
   VMContext* ctx = Processor::currentProcessor()->currentContext();
   CallFrame& frame = ctx->currentFrame();
   Function* func = ctx->currentFrame().m_function;
   ItemDict* dict = new ItemDict;
   int size = frame.m_paramCount < func->paramCount() ? frame.m_paramCount : func->paramCount();

   for( int i = 0; i < size; ++i )
   {
      String* name = new String( func->parameters().getNameById(i) );
      Item* param = ctx->param(i);
      dict->insert( FALCON_GC_HANDLE(name), *param );
   }

   value = FALCON_GC_HANDLE(dict);
}


static void get_name( const Class*, const String&, void* instance, Item& value )
{
   Function* func = static_cast<Function*>(instance);
   value = FALCON_GC_HANDLE( new String( func->name() ) );
}

static void get_fullname( const Class*, const String&, void* instance, Item& value )
{
   Function* func = static_cast<Function*>(instance);
   value = FALCON_GC_HANDLE( new String( func->fullName() ) );
}

static void get_location( const Class*, const String&, void* instance, Item& value )
{
   Function* func = static_cast<Function*>(instance);
   value = FALCON_GC_HANDLE( new String( func->locate() ) );
}

static void get_module( const Class*, const String&, void* instance, Item& value )
{
   static class ClassModule* clsModule = Engine::instance()->stdHandlers()->moduleClass();

   Function* func = static_cast<Function*>(instance);
   Module* owner = func->module();
   if( owner == 0 )
   {
      value.setNil();
   }
   else {
      value.setUser( clsModule, owner );
   }
}

static void get_methodOf( const Class*, const String&, void* instance, Item& value )
{
   static class Class* clsMeta = Engine::instance()->stdHandlers()->metaClass();

   Function* func = static_cast<Function*>(instance);
   Class* owner = func->methodOf();
   if( owner == 0 )
   {
      value.setNil();
   }
   else {
      value.setUser( clsMeta, owner );
   }
}


static void get_signature( const Class*, const String&, void* instance, Item& value )
{
   Function* func = static_cast<Function*>(instance);
   value = FALCON_GC_HANDLE( new String( func->signature() ) );
}


static void get_plist( const Class*, const String&, void* instance, Item& value )
{
   Function* func = static_cast<Function*>(instance);

   ItemArray* params = new ItemArray;
   const SymbolMap& vars = func->parameters();
   for( uint32 i = 0; i < vars.size(); ++i ) {
      params->append( FALCON_GC_HANDLE(new String(vars.getNameById(i) ) ) );
   }

   value = FALCON_GC_HANDLE(params);
}

static void get_caller( const Class*, const String&, void*, Item& value )
{
   Processor* prc = Processor::currentProcessor();
   if( prc == 0 )
   {
      value.setNil();
   }
   else
   {
      VMContext* ctx = prc->currentContext();
      if( ctx->callDepth() < 2 )
      {
         value.setNil();
      }
      else {
         const CallFrame& cf = ctx->callerFrame(1);
         if(cf.m_bMethodic)
         {
            value = cf.m_self;
            value.methodize(cf.m_function);
         }
         else {
            value.setFunction(cf.m_function);
         }
      }
   }
}

static void get_invoker( const Class*, const String&, void*, Item& value )
{
   Processor* prc = Processor::currentProcessor();
   if( prc == 0 )
   {
      value.setNil();
   }
   else
   {
      VMContext* ctx = prc->currentContext();
      if( ctx->callDepth() < 2 )
      {
         value.setNil();
      }
      else {
         const CallFrame& cf = prc->currentContext()->callerFrame(1);
         if(cf.m_bMethodic)
         {
            value = cf.m_self;
         }
         else {
            value.setNil();
         }
      }
   }
}


namespace CFunction {
/*#
 @method parameter Function
 @brief (static) Gets the nth parameter from the current function call.
 @param nth The nth position (zero based), or the name of the parameter.
 @return The value stored for the given parameter.
 @raise AccessError if the value is out of range.
 */
FALCON_DECLARE_FUNCTION(parameter, "nth:N|S")
void Function_parameter::invoke( VMContext* ctx, int32 pCount )
{
   if( pCount <= 0 )
   {
      throw paramError(__LINE__, SRC );
   }

   Item iParam = *ctx->param(0);
   // return to the previous function
   ctx->returnFrame();
   fassert( ctx->callDepth() >= 0 );
   CallFrame& frame = ctx->currentFrame();

   if( iParam.isOrdinal() )
   {
      uint32 id = (uint32) iParam.forceInteger();
      if ( id >= frame.m_paramCount )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra("Parameter number out of range"));
      }

      ctx->topData() = *ctx->param(id);
   }
   else if( iParam.isString() )
   {
      const String& name = *iParam.asString();
      SymbolMap& vars = frame.m_function->parameters();
      int32 paramId = vars.find(name);
      if( paramId < 0 )
      {
         throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra(String("Unknown parameter ") + name ));
      }

      if( paramId < (int32) frame.m_paramCount )
      {
         ctx->topData() = *ctx->param(paramId);
      }
      // else, the nil created at previous return frame is ok
   }
   else {
      // we have not a number nor a string.
      throw paramError(__LINE__, SRC );
   }
}

// I want to try the Ext function constructor...
static void mth_ctor( VMContext* ctx, int32 )
{
   String name;
   Item* i_name = ctx->param(0);
   Item* i_proto = ctx->param(1);

   if( i_name != 0 && ! i_name->isNil() )
   {
      if( i_name->isString() )
      {
         name = *i_name->asString();
      }
      else {
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("[S],[S]"));
      }
   }

   Function* func = new SynFunc(name);
   if( i_proto != 0 && ! i_proto->isNil() )
   {
      if( i_proto->isString() )
      {
         const String& sign = *i_proto->asString();
         if( ! func->parseDescription(sign) )
         {
            delete func;
            throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra("Invalid signature"));
         }
      }
      else {
         delete func;
         throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("[S],[S]"));
      }
   }

   ctx->pushData(FALCON_GC_HANDLE(func));
}


/*#
 @method call Funciton
 @brief Invokes the given function passing the parameters from an array.
 @optparam params Array of parameters to be sent to the function.
 @return The value returned by the invoked function.

 This function can be used to efficiently invoke a function for which
 the parameters have been stored in a an array.

 The called function replaces this method in the call stack, as if
 it was directly called.

 The following calls are equivalent:
 @code
    function test(a,b)
       > "A: ", a
       > "B: ", b
    end

    test("a","b")
    [test, "a"]("b")
    test.call( ["a","b"])
 @endcode

 @see passvp
 */
FALCON_DECLARE_FUNCTION(call, "params:[A]")
void Function_call::invoke( VMContext* ctx, int32 )
{
   Item* iParams = ctx->param(0);
   if(iParams != 0 && ! iParams->isArray())
   {
      throw paramError(__LINE__, SRC);
   }

   ItemArray* ir = iParams == 0 ? 0 : iParams->asArray();
   Item self = ctx->self();
   ctx->returnFrame();
   ctx->popData();

   if( ir == 0 )
   {
      ctx->callerLine(__LINE__+1);
      ctx->callItem(self);
   }
   else {
      ItemArray local;
      // mutlitasking wise...
      local.copyOnto( *ir );
      ctx->callerLine(__LINE__+1);
      ctx->callItem( self, local.length(), local.elements() );
   }
}


class PStepRedo: public PStep
{
public:
   PStepRedo() { apply = apply_; }
   virtual ~PStepRedo() {}
   virtual void describeTo( String& tgt ) const { tgt = "PStepRedo"; }

   static void apply_(const PStep*, VMContext* ctx )
   {
      int32& seqId = ctx->currentCode().m_seqId;
      Item* curParam = ctx->param(seqId);
      // no more params?
      if( curParam == 0 )
      {
         ctx->popCode();
         return;
      }

      ++seqId;
      Item param = *curParam;
      CallFrame& cf = ctx->currentFrame();
      uint32 pc = cf.m_function->paramCount();

      ctx->pushData(cf.m_function);
      for( uint32 i = 0; i + 1 < pc; ++i )
      {
         Item temp = *ctx->param(i);
         ctx->pushData(temp);
      }

      ctx->pushData(param);
      ClosedData* closed =  cf.m_closure;
      if( cf.m_bMethodic )
      {
         ctx->callInternal(cf.m_function, pc, cf.m_self );
      }
      else {
         ctx->callInternal(cf.m_function, pc );
      }

      // repeat the closure if necessary.
      ctx->currentFrame().m_closure = closed;
   }
};

/*#
 @method redo Funciton
 @brief (static) Invokes the current function passing the variable parameters one at a time.

 This method invokes the current function, and passes each non-declared parameters
 to it one at a time.

 It is equivalent to the following code:
 @code
 function test( param )
    // do things...

   // equivalent to Function.redo()
    for p in Function.vp
       fself(p)
    end
 end
 @endcode

 For instance, a function printing an arbitrary list of terms can be writen as:

 @code
 function printList()
    for elem in Function.vp
       > "Elem is: ", elem
    end
 end

 // or
 function printList( elem )
    > "Elem is: ", elem
    Function.redo()
 end
 @endcode

 The second form saves a for-loop and the creation of the variable parameter array,
 resulting widely more efficient. Also, it is possible to write the function simply
 as if it had a single parameter, and then just add Function.redo() that will be nearly
 no-op if the function is actually called with a single parameter.

 If the function has more than one declared parameter, then all the declared parameters but
 the last one are passed unchanged, and the last one assumes the value of each variable
 parameter in turn. For example:

 @code
 function printWithPrompt( p, elem )
    > p, ": ", elem
    function.redo()
 end

 printWithPrompt( "Elem is", 1, 2, 3, 4 )
 @endcode

 In the above code, the @b parameter rest unchanged, while @b elem becomes 1, 2, 3 and 4
 in each subsequent invocation.

 @note If the current function is a method, the self item is repeated as if calling
 @b self.method.

 @see passvp
 */
FALCON_DECLARE_FUNCTION_EX(redo, "", PStepRedo m_stepRedo; )
void Function_redo::invoke( VMContext* ctx, int32 )
{
   // we abandon the frame immediately, as we operate in the caller frame.
   ctx->returnFrame();

   CallFrame& frame = ctx->currentFrame();
   uint32 pdef = frame.m_function->paramCount();
   uint32 pc = ctx->currentFrame().m_paramCount;

   if( pc <= pdef ){
      return;
   }

   ctx->pushCode( &m_stepRedo );
   ctx->currentCode().m_seqId = pdef;
   // let the VM handle the rest.
}


}


ClassFunction::ClassFunction(ClassMantra* parent):
   ClassMantra("Function", FLC_CLASS_ID_FUNC )
{
   addProperty("current", &get_current, 0, true );
   addProperty("pcount", &get_pcount, 0, true );
   addProperty("params", &get_params, 0, true );
   addProperty("vp", &get_vp, 0, true );
   addProperty("vcount", &get_vcount, 0, true );
   addProperty("pdict", &get_pdict, 0, true );
   addProperty("caller", &get_caller, 0, true );
   addProperty("invoker", &get_invoker, 0, true );

   addProperty("name", &get_name );
   addProperty("fullname", &get_fullname );
   addProperty("location", &get_location );
   addProperty("module", &get_module );
   addProperty("methodOf", &get_methodOf );
   addProperty("signature", &get_signature );
   addProperty("plist", &get_plist );

   addMethod(new CFunction::Function_parameter, true);
   addMethod(new CFunction::Function_call );
   addMethod(new CFunction::Function_redo, true );
   setConstuctor( &CFunction::mth_ctor, "name:[S]");

   setParent(parent);
}


ClassFunction::~ClassFunction()
{
}


void* ClassFunction::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassFunction::describe( void* instance, String& target, int, int ) const
{
   Function* func = static_cast<Function*>(instance);
   target = func->name() + "(" + func->getDescription() +")";
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
   ctx->topData() = FALCON_GC_HANDLE( ret );
}

void ClassFunction::op_iter( VMContext* ctx, void* instance ) const
{
   ctx->pushData(Item(this, instance));
}

void ClassFunction::op_next( VMContext* ctx, void* instance ) const
{
   //self (function) is already on top of the stack.
   Function* func = static_cast<Function*>(instance);
   ctx->pushData(Item(this, func));
   ctx->callInternal(func,0);
}

}

/* end of classfunction.cpp */
