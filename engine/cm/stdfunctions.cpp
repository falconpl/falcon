/*
   FALCON - The Falcon Programming Language.
   FILE: stdfunctions.h

   Falcon core module -- Standard core module functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 22 Jan 2013 11:24:20 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/stdfunctions.cpp"

#include <falcon/trace.h>
#include <falcon/falcon.h>

#include <falcon/autocstring.h>
#include <falcon/cm/stdfunctions.h>
#include <falcon/stdstreams.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/sys.h>
#include <falcon/stdhandlers.h>

#include <falcon/cm/siter.h>
#include <falcon/itemarray.h>

#include <falcon/error.h>
#include <falcon/stderrors.h>
#include <falcon/process.h>

namespace Falcon {
namespace Ext {

/*#
   @function sleep
   @brief Put the current coroutine at sleep for some time.
   @param time Time, in seconds and fractions, that the caller wishes to sleep.

   This function declares that the current coroutines is not willing to proceed at
   least for the given time. The VM will swap out the caller until the time has
   elapsed, and will make it eligible to run again after the given time lapse.

   The parameter may be a floating point number if a pause shorter than a second is
   required.

   @note As this call is performed, any critical section is abandoned, and acquired
   shared resources are signaled.
*/

FALCON_DEFINE_FUNCTION_P(sleep)
{
   TRACE1( "-- called with %d params", pCount );

   // all the evaluation happens in the
   if( pCount < 1 || ! ctx->param(0)->isOrdinal() ) {
      throw paramError();
   }

   numeric to = ctx->param(0)->forceNumeric();
   ctx->sleep( (int64)(to * 1000) );
   ctx->returnFrame();
}

/*#
   @function rest
   @brief Put the current coroutine at sleep for some time.
   @param time Time, in milliseconds, that the caller wishes to sleep.

   This function declares that the current context is not willing to proceed at
   least for the given time. The VM will swap out the caller until the time has
   elapsed, and will make it eligible to run again after the given time lapse.

   @note this is equivalent to @a sleep but the sleep @b time is expressed in
   milliseconds.

   @note As this call is performed, any critical section is abandoned, and aquired
   shared resources are signaled.
*/

FALCON_DEFINE_FUNCTION_P(rest)
{
   TRACE1( "-- called with %d params", pCount );

   // all the evaluation happens in the
   if( pCount < 1 || ! ctx->param(0)->isOrdinal() ) {
      throw paramError();
   }

   int64 to = ctx->param(0)->forceInteger();
   ctx->sleep( to );
   ctx->returnFrame();
}

/**
 * Moved in sys
   @function epoch
   @ingroup general_purpose
   @brief Returns the number of seconds since the "epoch" (1 Jan 1970).
   @return An integer number of seconds.
*/
/*FALCON_DEFINE_FUNCTION_P1(epoch)
{
   MESSAGE1( "-- called with 0 params" );
   int64 ep = Sys::_epoch();
   ctx->returnFrame(ep);
}
*/
/*#
   @function seconds
   @ingroup general_purpose
   @brief Returns the number of seconds and milliseconds from day, activity or program start.
   @return The number of seconds and fractions of seconds in a floating point value.

   Actually, this function returns a floating point number which represents seconds
   and fraction of seconds elapse since a conventional time. This function is mainly
   meant to be used to take intervals of time inside the script,
   with a millisecond precision.
*/

FALCON_DEFINE_FUNCTION_P1(seconds)
{
   MESSAGE1( "-- called with 0 params" );
   numeric ep = Sys::_milliseconds()/1000.0;
   ctx->returnFrame(ep);
}

/*#
   @function quit
   @ingroup general_purpose
   @brief Terminates the current process.
   @optparam value The process termination value.

   This function terminates the current virtual machine process as soon
   as possible. This doesn't force the underlying O/S into terminating
   the host process.
*/

FALCON_DEFINE_FUNCTION_P(quit)
{
   TRACE( "-- called with %d params", pCount );

   if(pCount > 0)
   {
      ctx->process()->setResult( * ctx->param(0) );
   }

   ctx->process()->terminate();
   ctx->returnFrame();
}

/*#
   @function advance
   @ingroup general_purpose
   @param collection the collection being traversed.
   @brief Advnaces an automatic iterator in the current rule context.
   @return An item from the collection.

*/

FALCON_DEFINE_FUNCTION_P(advance)
{
   TRACE1( "-- called with %d params", pCount );
   if( pCount < 1 ) {
      throw paramError();
   }

   SIterCarrier* ic;
   Class* iterClass=0;

   if( ctx->readInit().isNil() )
   {
      iterClass = module()->getClass("Iterator");
      fassert( iterClass != 0 );
      ic = new SIterCarrier( *ctx->param(0) );

      ctx->writeInit(FALCON_GC_STORE(iterClass, ic));
   }
   else {
      const Item& initItem = ctx->readInit();
      void* data = 0;
      initItem.asClassInst( iterClass, data );
      fassert( iterClass == module()->getClass("Iterator") );
      ic = static_cast<SIterCarrier*>(data);
   }

   // change into the class method, and use its return frames.
   ctx->param(0)->setBoolean(false);
   static_cast<ClassSIter*>(iterClass)->invokeDirectNextMethod(ctx, ic, pCount);
}

/*#
   @function int
   @brief Converts the given parameter to integer.
   @param item The item to be converted
   @return An integer value.
   @raise ParseError in case the given string cannot be converted to an integer.
   @raise MathError if a given floating point value is too large to be converted to an integer.

   Integer values are just copied. Floating point values are converted to long integer;
   in case they are too big to be represented a RangeError is raised.
   Strings are converted from base 10. If the string cannot be converted,
   or if the value is anything else, a MathError instance is raised.
*/

FALCON_DEFINE_FUNCTION_P(int)
{
   TRACE1( "int -- called with %d params", pCount );
   if( pCount < 1 ) {
      throw paramError( __LINE__, SRC );
   }

   Item* i_param = ctx->param(0);
   if( i_param->isInteger() ) {
      ctx->returnFrame(*i_param);
   }
   else if( i_param->isNumeric() )
   {
      numeric nval = i_param->asNumeric();
      if ( nval > 9.223372036854775808e18 || nval < -9.223372036854775808e18 )
      {
         throw new MathError( ErrorParam( e_domain, __LINE__ ).origin( ErrorParam::e_orig_runtime ) );
      }

      ctx->returnFrame( (int64) nval );
   }
   else if( i_param->isString() )
   {
      String* str = i_param->asString();
      int64 num = 0;
      double nval = 0;
      if( str->parseInt( num, 0 ) )
      {
         ctx->returnFrame(num);
      }
      else if (str->parseDouble(nval, 0))
      {
         if ( nval > 9.223372036854775808e18 || nval < -9.223372036854775808e18 )
         {
            throw new MathError( ErrorParam( e_domain, __LINE__ ).origin( ErrorParam::e_orig_runtime ) );
         }
         ctx->returnFrame( (int64) nval );
      }
      else {
         throw FALCON_SIGN_XERROR(ParamError, e_param_type, .extra("Not a number"));
      }
   }
   else {
      throw paramError( __LINE__, SRC );
   }
}

FALCON_DEFINE_FUNCTION_P(numeric)
{
   TRACE1( "numeric -- called with %d params", pCount );
   if( pCount < 1 ) {
      throw paramError( __LINE__, SRC );
   }

   Item* i_param = ctx->param(0);
   if( i_param->isInteger() ) {
      ctx->returnFrame((numeric) i_param->asInteger() );
   }
   else if( i_param->isNumeric() )
   {
      ctx->returnFrame(i_param->asNumeric());
   }
   else if( i_param->isString() )
   {
      String* str = i_param->asString();
      double dbl = 0.0;
      if( str->parseDouble(dbl,0) )
      {
         ctx->returnFrame(dbl);
      }
      else {
         throw FALCON_SIGN_XERROR(ParamError, e_param_type, .extra("Not a number"));
      }
   }
   else {
      throw paramError( __LINE__, SRC );
   }
}

/*#
   @function input
   @inset core_basic_io
   @brief Get some text from the user (standard input stream).

   Reads a line from the standard input stream and returns a string
   containing the read data. This is mainly meant as a test/debugging
   function to provide the scripts with minimal console based user input
   support. When in need of reading lines from the standard input, prefer the
   readLine() method of the input stream object.

   This function may also be overloaded by embedders to provide the scripts
   with a common general purpose input function, that returns a string that
   the user is queried for.
*/

FALCON_DEFINE_FUNCTION_P(input)
{
   TRACE1( "input -- called with %d params", pCount );
   (void) pCount;

   Process* proc = ctx->process();
   String* str = new String;
   try
   {
      proc->textIn()->readLine(*str, 4096);
      ctx->returnFrame(FALCON_GC_HANDLE(str));
   }
   catch( ... )
   {
      delete str;
      throw;
   }

}

/*#
   @function passvp
   @inset varparams_support
   @brief Returns all the undeclared parameters, or passes them to a callable item
   @optparam citem Callable item on which to pass the parameters.
   @return An array containing unnamed parameters, or the return value \b citem.

   This function returns all the parameters passed to this function but not declared
   in its prototype (variable parameters) in an array.

   If the host function doesn't receive any extra parameter, this function returns
   an empty array. This is useful in case the array is immediately added to a direct
   call. For example:

   @code
   function receiver( a, b )
      > "A: ", a
      > "B: ", b
      > "Others: ", passvp().describe()
   end

   receiver( "one", "two", "three", "four" )
   @endcode

   If @b citem is specified, the function calls citem passing all the extra parameters
   to it. For example:

   @code
   function promptPrint( prompt )
      passvp( .[printl prompt] )
   end

   promptPrint( "The prompt: ", "arg1", " ", "arg2" )
   @endcode
*/
FALCON_DEFINE_FUNCTION_P(passvp)
{
   if ( pCount > 0 )
   {
      Item callee = *ctx->param(0);

      ctx->returnFrame();
      CallFrame& frame = ctx->currentFrame();
      Function* func = frame.m_function;

      ctx->pushData(callee);
      for( int i = func->paramCount(); i < ctx->paramCount(); ++i )
      {
         ctx->pushData( *ctx->param(i) );
      }
      int callCount = ctx->paramCount() - func->paramCount();
      if( callCount < 0 )
      {
         callCount = 0;
      }

      Class* calleeClass;
      void* calleeData;
      callee.forceClassInst(calleeClass, calleeData);
      calleeClass->op_call( ctx, callCount, calleeData );

   }
   else
   {
      ctx->returnFrame();
      CallFrame& frame = ctx->currentFrame();
      Function* func = frame.m_function;
      ItemArray* retval = new ItemArray;
      int32 i = func->paramCount();
      int32 max = ctx->paramCount();

      for( ; i < max; ++i )
      {
         retval->append( *ctx->param(i) );
      }
      ctx->topData() = FALCON_GC_HANDLE(retval);
   }
}



inline static void genNext( VMContext* ctx )
{
   Class* cls = 0;
   void* data = 0;
   ctx->param(1)->forceClassInst(cls, data);
   cls->op_next(ctx,data);
}

/*#
   @function map
   @inset functional_support
   @brief Changes all the items in the source data according to a mapping function.
   @param mapper Function or code receiving the source data items.
   @param data A sequence (array, generator, range etc.).
   @return An array containing the mapped data.
   @raise AccessError if @b data is not iterable.

   The @b mapper is called iteratively for every item in the @b data; its return value is added to
   an array that is then returned.

   The following code generates the squares between 1-3.
   @code
   x = map( {(v) v**2}, [1,2,3] )
   > x.describe()                // [1,4,9]
   @endcode

   This function is equivalent to the following accumulator operator:
   @code
   ^[ data ] mapper => []
   @endcode

   but, it is slightly more efficient.

   @note If the mapper invokes a @b return @b break statement, the mapping is interrupted.
*/

FALCON_DEFINE_FUNCTION_P1(map)
{
   // This is called after op_next generates the next item.
   class PStepGetNext: public PStep {
   public:
      PStepGetNext(){ apply = apply_;}
      virtual ~PStepGetNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_map::PStepGetNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         ctx->popCode();
         // if not, add the item to the array.
         // below us, we have the check filter code.
         ctx->local(0)->asArray()->append(ctx->topData());
         ctx->popData();

         // call the next operator again
         genNext(ctx);
      }
   };
   static PStepGetNext s_stepGetNext;

   class PStepProcessNext: public PStep {
   public:
      PStepProcessNext(){ apply = apply_;}
      virtual ~PStepProcessNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_map::PStepProcessNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         // if not, call the mapper function
         ctx->pushCode( &s_stepGetNext );
         Item temp = ctx->topData();
         ctx->popData();
         ctx->callItem(*ctx->param(0), 1, &temp);
      }
   };
   static PStepProcessNext s_stepProcessNext;


   // This is invoked after op_iter is called.
   class PStepBeginIter: public PStep {
   public:
      PStepBeginIter(){ apply = apply_;}
      virtual ~PStepBeginIter() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_map::PStepBeginIter"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // if op_iter returns break, we throw
         if( ctx->topData().isBreak() ) {
            throw FALCON_SIGN_ERROR( AccessError, e_invalid_iter );
         }

         // prepare the return array.
         *ctx->local(0) = FALCON_GC_HANDLE(new ItemArray());

         // step into getting the next item.
         ctx->resetCode( &s_stepProcessNext );
         genNext(ctx);
      }
   };
   static PStepBeginIter s_stepBeginIter;

   // add a  local space, we'll need it
   ctx->addLocals(1);

   Item* iItem = ctx->param(0);
   Item* iParams = ctx->param(1);
   if( iItem == 0 || iParams == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   // prepare the code that will interpret op_iter result
   ctx->pushCode( &s_stepBeginIter );

   // prepare the op_iter framework
   ctx->pushData( *iParams );
   Class* cls = 0;
   void* data = 0;
   iParams->forceClassInst(cls, data);
   cls->op_iter(ctx,data);
}

/*#
   @function reduce
   @inset functional_support
   @brief Uses the values in a given sequence and iteratively calls a reductor function to extract a single result.
   @param reducer A function or Sigma to reduce the array.
   @param data A sequence of arbitrary items.
   @optparam initial Optional startup value for the reduction.
   @return The reduced result.

   The reductor is a function receiving two values as parameters. The first value is the
   previous value returned by the reductor, while the second one is an item iteratively
   taken from the origin array. If a startup value is given, the first time the reductor
   is called that value is provided as its first parameter, otherwise the first two items
   from the array are used in the first call. If the collection is empty, the initial_value
   is returned instead, and if is not given, nil is returned. If a startup value is not given
   and the collection contains only one element, that element is returned.

   Some examples:
   @code
   > reduce( {a,b=> a+b}, [1,2,3,4])       // sums 1 + 2 + 3 + 4 = 10
   > reduce( {a,b=> a+b}, [1,2,3,4], -1 )  // sums -1 + 1 + 2 + 3 + 4 = 9
   > reduce( {a,b=> a+b}, [1] )            // never calls lambda, returns 1
   > reduce( {a,b=> a+b}, [], 0 )          // throws
   > reduce( {a,b=> a+b}, [] )             // throws
   @endcode

   Items in the collection are treated literally (not evaluated).
*/
FALCON_DEFINE_FUNCTION_P1(reduce)
{
   // This is called after op_next generates the next item.
   class PStepGetNext: public PStep {
   public:
      PStepGetNext(){ apply = apply_;}
      virtual ~PStepGetNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_reduce::PStepGetNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         ctx->popCode();
         *ctx->local(0) = ctx->topData();
         ctx->popData();
         // call the next operator again
         genNext(ctx);
      }
   };
   static PStepGetNext s_stepGetNext;

   class PStepInvokeReduce: public PStep {
   public:
      PStepInvokeReduce(){ apply = apply_;}
      virtual ~PStepInvokeReduce() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_reduce::PStepInvokeReduce"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         // if not, call the mapper function
         ctx->pushCode( &s_stepGetNext );
         Item params[2];
         params[0] = *ctx->local(0);
         params[1] = ctx->topData();
         ctx->popData();
         // we need to keep the top data.
         ctx->callItem(*ctx->param(0), 2, params);
      }
   };
   static PStepInvokeReduce s_stepInvokeReduce;

   // This is invoked after op_iter is called.
   class PStepGenSecond: public PStep {
   public:
      PStepGenSecond(){ apply = apply_;}
      virtual ~PStepGenSecond() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_reduce::PStepGenSecond"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // if op_iter returns break, we throw
         if( ctx->topData().isBreak() ) {
            throw FALCON_SIGN_XERROR( AccessError, e_invalid_iter, .extra("The series din't have enough items to be reduced") );
         }
         *ctx->local(0) = ctx->topData();
         ctx->popData();
         // step into getting the next item.
         ctx->resetCode( &s_stepInvokeReduce );
         genNext(ctx);
      }
   };
   static PStepGenSecond s_stepGenSecond;

   // This is invoked after op_iter is called.
   class PStepGenFirst: public PStep {
   public:
      PStepGenFirst(){ apply = apply_;}
      virtual ~PStepGenFirst() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_reduce::PStepGenFirst"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // if op_iter returns break, we throw
         if( ctx->topData().isBreak() ) {
            throw FALCON_SIGN_ERROR( AccessError, e_invalid_iter );
         }

         if( ctx->paramCount() < 3 )
         {
            // step into getting the next item.
            ctx->resetCode( &s_stepGenSecond );
         }
         else
         {
            *ctx->local(0) = *ctx->param(2);
            ctx->resetCode( &s_stepInvokeReduce );
         }

         // step into getting the next item.
         genNext(ctx);
      }
   };
   static PStepGenFirst s_stepGenFirst;

   // add a  local space, we'll need it
   ctx->addLocals(1);

   Item* iItem = ctx->param(0);
   Item* iParams = ctx->param(1);
   if( iItem == 0 || iParams == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   // prepare the op_iter framework
   ctx->pushCode( &s_stepGenFirst );
   ctx->pushData( *iParams );
   Class* cls = 0;
   void* data = 0;
   iParams->forceClassInst(cls, data);
   cls->op_iter(ctx,data);
}


/*#
   @function filter
   @inset functional_support
   @brief Filters sequence using a filter function.
   @param ffunc A callable item used to filter the array.
   @param sequence A sequence of arbitrary items.
   @return The filtered sequence.

   ffunc is called iteratively for every item in the collection, which is passed as a parameter to it.
   If the call returns true, the item is added to the returned array; if it returns false,
   the item is not added.

   Items in the collection are treated literally (not evaluated).
*/
FALCON_DEFINE_FUNCTION_P1(filter)
{
   // This is called after op_next generates the next item.
   class PStepGetNext: public PStep {
   public:
      PStepGetNext(){ apply = apply_;}
      virtual ~PStepGetNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_filter::PStepGetNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         ctx->popCode();

         bool toGet = ctx->topData().isTrue();
         ctx->popData();
         if( toGet )
         {
            // if not, add the item to the array.
            // below us, we have the check filter code.
            ctx->local(0)->asArray()->append(ctx->topData());
         }

         ctx->popData();
         // call the next operator again
         genNext(ctx);
      }
   };
   static PStepGetNext s_stepGetNext;

   class PStepProcessNext: public PStep {
   public:
      PStepProcessNext(){ apply = apply_;}
      virtual ~PStepProcessNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_filter::PStepProcessNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         // if not, call the mapper function
         ctx->pushCode( &s_stepGetNext );
         Item temp = ctx->topData();
         // we need to keep the top data.
         ctx->callItem(*ctx->param(0), 1, &temp);
      }
   };
   static PStepProcessNext s_stepProcessNext;


   // This is invoked after op_iter is called.
   class PStepBeginIter: public PStep {
   public:
      PStepBeginIter(){ apply = apply_;}
      virtual ~PStepBeginIter() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_filter::PStepBeginIter"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // if op_iter returns break, we throw
         if( ctx->topData().isBreak() ) {
            throw FALCON_SIGN_ERROR( AccessError, e_invalid_iter );
         }

         // prepare the return array.
         *ctx->local(0) = FALCON_GC_HANDLE(new ItemArray());

         // step into getting the next item.
         ctx->resetCode( &s_stepProcessNext );
         genNext(ctx);
      }
   };
   static PStepBeginIter s_stepBeginIter;

   // add a  local space, we'll need it
   ctx->addLocals(1);

   Item* iItem = ctx->param(0);
   Item* iParams = ctx->param(1);
   if( iItem == 0 || iParams == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   // prepare the code that will interpret op_iter result
   ctx->pushCode( &s_stepBeginIter );

   // prepare the op_iter framework
   ctx->pushData( *iParams );
   Class* cls = 0;
   void* data = 0;
   iParams->forceClassInst(cls, data);
   cls->op_iter(ctx,data);
}


/*#
   @function cascade
   @inset functional_support
   @brief Concatenate a set of callable items so to form a single execution unit.
   @param callList Sequence of callable items.
   @optparam ... Optional parameters to be passed to the first callable item.
   @return The return value of the last callable item.

   This function executes a set of callable items passing the parameters it receives
   beyond the first one to the first  item in the list; from there on, the return value
   of the previous call is fed as the sole parameter of the next call. In other words,
   @code
      cascade( [F1, F2, ..., FN], p1, p2, ..., pn )
   @endcode
   is equivalent to
   @code
      FN( ... F2( F1( p1, p2, ..., pn ) ) ... )
   @endcode

   A function may declare itself "uninterested" to insert its value in the cascade
   by returning an out-of-band item. In that case, the return value is ignored and the same parameter
   it received is passed on to the next calls and eventually returned.

   Notice that the call list is not evaluated in functional context; it is just a list
   of callable items. To evaluate the list, or part of it, in functional context, use
   the eval() function.

   A simple example usage is the following:
   @code
      function square( a )
         return a * a
      end

      function sqrt( a )
         return a ** 0.5
      end

      cascade_abs = [cascade, [square, sqrt] ]
      > cascade_abs( 2 )      // 2
      > cascade_abs( -4 )     // 4
   @endcode

   Thanks to the possibility to prevent insertion of the return value in the function call sequence,
   it is possible to program "interceptors" that will catch the progress of the sequence without
   interfering:

   @code
      function showprog( v )
         > "Result currently ", v
        return oob(nil)
      end

      // define sqrt and square as before...
      cascade_abs = [cascade, [square, showprog, sqrt, showprog] ]
      > "First process: ", cascade_abs( 2 )
      > "Second process: ", cascade_abs( -4 )
   @endcode

   If the first function of the list declines processing by returning an oob item, the initial parameters
   are all passed to the second function, and so on till the last call.

   For example:

   @code
      function whichparams( a, b )
         > "Called with ", a, " and ", b
         return oob(nil)
      end

      csq = [cascade, [ whichparams, {a,b=> a*b} ]
      > csq( 3, 4 )
   @endcode

   Here, the first function in the list intercepts the parameters but, as it doesn't
   accepts them, they are both passed to the second in the list.

   @see oob
*/
FALCON_DEFINE_FUNCTION_P1(cascade)
{
   // This is called after op_next generates the next item.
   class PStepGetNext: public PStep {
   public:
      PStepGetNext(){ apply = apply_;}
      virtual ~PStepGetNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_cascade::PStepGetNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         // remove this code, PStepFuncResult is now in charge.
         ctx->popCode();

         Item func = ctx->topData();
         ctx->pushData( *ctx->local(0) );
         ctx->callInternal(func, 1);
      }
   };
   static PStepGetNext s_stepGetNext;

   class PStepFuncResult: public PStep {
   public:
      PStepFuncResult(){ apply = apply_;}
      virtual ~PStepFuncResult() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_cascade::PStepFuncResult"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // save the result and ask for the next operator
         *ctx->local(0) = ctx->topData();
         ctx->popData();

         ctx->pushCode( &s_stepGetNext );
         Class* cls = 0;
         void* data = 0;
         ctx->param(0)->forceClassInst(cls, data);
         cls->op_next(ctx, data);
      }
   };
   static PStepFuncResult s_stepFuncResult;


   // processes the first result of the iteration (use all the parameters)
   class PStepProcessFirst: public PStep {
   public:
      PStepProcessFirst(){ apply = apply_;}
      virtual ~PStepProcessFirst() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_cascade::PStepProcessFirst"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // Empty sequence? -- return nil
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame();
            return;
         }

         // if not, call the mapper function
         ctx->resetCode( &s_stepFuncResult );
         Item func = ctx->topData();

         // we need to keep the top data.
         for( int32 i = 1; i < ctx->paramCount(); ++i )
         {
            ctx->pushData( *ctx->param(i) );
         }
         ctx->callInternal(func, ctx->paramCount()-1 );
      }
   };
   static PStepProcessFirst s_stepProcessFirst;


   // This is invoked after op_iter is called.
   class PStepBeginIter: public PStep {
   public:
      PStepBeginIter(){ apply = apply_;}
      virtual ~PStepBeginIter() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_cascade::PStepBeginIter"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // if op_iter returns break, we throw
         if( ctx->topData().isBreak() ) {
            throw FALCON_SIGN_ERROR( AccessError, e_invalid_iter );
         }

         // step into getting the next item.
         ctx->resetCode( &s_stepProcessFirst );

         Class* cls = 0;
         void* data = 0;
         ctx->param(0)->forceClassInst(cls, data);
         cls->op_next(ctx,data);
      }
   };
   static PStepBeginIter s_stepBeginIter;

   // add a  local space, we'll need it to store function results
   ctx->addLocals(1);

   Item* iItem = ctx->param(0);
   if( iItem == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   // prepare the code that will interpret op_iter result
   ctx->pushCode( &s_stepBeginIter );

   // prepare the op_iter framework
   Class* cls = 0;
   void* data = 0;
   iItem->forceClassInst(cls, data);
   ctx->pushData(*iItem);
   cls->op_iter(ctx,data);
}

/*#
  @function perform
  @brief (ETA) Invokes its parameters one after another.
  @param ... ETA expressions to be invoked.
  @return The result of the evaluation of the last expression.

  This function invokes all the expressions passed as ETA parameters
  one after another.

  If one expression yields a return break value, the computation is
  interrupted and the previous evaluation result is returned.
 */

FALCON_DEFINE_FUNCTION_P1(perform)
{
   // This is called after op_next generates the next item.
   class PStepGetNext: public PStep {
   public:
      PStepGetNext(){ apply = apply_;}
      virtual ~PStepGetNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_perform::PStepGetNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         *ctx->local(0) = ctx->topData();
         ctx->popData();
         // remove this code, PStepFuncResult is now in charge.
         CodeFrame& cf = ctx->currentCode();
         if( cf.m_seqId >= ctx->paramCount() )
         {
            ctx->returnFrame(*ctx->local(0));
            return;
         }

         Item* expr = ctx->param(cf.m_seqId++);
         TreeStep* ts = static_cast<TreeStep*>(expr->asInst());
         ctx->pushCode( ts );
      }
   };
   static PStepGetNext s_stepNext;

   // prepare the code that will interpret op_iter result
   ctx->addLocals(1);
   ctx->pushData(Item());
   ctx->stepIn( &s_stepNext );
}


/*#
  @function ffirstOf
  @brief (ETA) Evaluates the parameters and return the first non-nil result.
  @param ... Expressions to be evaluated.
  @return The result of the evaluation of the first non-nil expression, or nil.

  This function invokes all the expressions passed as ETA parameters
  one after another. Once the evaluated expression yields a non-nil value,
  that one is returned.

  If none of the expressions passed as paremters evaluates to a non-nil value,
  nil is returned.

  @notice As the expressions in the parameters are evaluated, this function
  is not well suited to return the generic first non-nil value. Use the
  non-eta function @a firstOf for that.
 */

FALCON_DEFINE_FUNCTION_P1(ffirstOf)
{
   // This is called after op_next generates the next item.
   class PStepGetNext: public PStep {
   public:
      PStepGetNext(){ apply = apply_;}
      virtual ~PStepGetNext() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_firstOf::PStepGetNext"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // are we done?
         if( ! ctx->topData().isNil() )
         {
            ctx->returnFrame(ctx->topData());
         }
         else {
            // remove this code, PStepFuncResult is now in charge.
            CodeFrame& cf = ctx->currentCode();
            if( cf.m_seqId >= ctx->paramCount() )
            {
               ctx->returnFrame();
            }
            else {
               Item* expr = ctx->param(cf.m_seqId++);
               TreeStep* ts = static_cast<TreeStep*>(expr->asInst());
               ctx->pushCode( ts );
            }
         }
      }
   };
   static PStepGetNext s_stepNext;

   // prepare the code that will interpret op_iter result
   ctx->pushData(Item());
   ctx->stepIn( &s_stepNext );
}

/*#
  @function firstOf
  @brief Returns the first non-nil parameter.
  @param ... Values to be verified.
  @return The first non-nil paramter, or nil.

  This function returns the first non-nil parameter among the given ones.
  If the function is called without parameters, or if all the parameters
  are nil values, it returns nil.
*/

FALCON_DEFINE_FUNCTION_P(firstOf)
{
   for( int i = 0; i < pCount; ++i )
   {
      fassert( ctx->param(i) != 0 );
      const Item& value = *ctx->param(i);
      if( ! value.isNil() )
      {
         ctx->returnFrame(value);
         return;
      }
   }

   // couldn't find it.
   ctx->returnFrame();
}


// NOTICE: for 1.0 alpha
// left currently undocumented.
// TODO: Check this out and document it if useful.

FALCON_DEFINE_FUNCTION_P(ffor)
{
   class PStepCheckLimit: public PStep {
   public:
      PStepCheckLimit(){ apply = apply_;}
      virtual ~PStepCheckLimit() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_ffor::PStepCheckLimit"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // save the result and ask for the next operator
         if( ! ctx->topData().isTrue() )
         {
            ctx->returnFrame();
            return;
         }

         ctx->popCode();
         ctx->popData();
         // below us there's s_stepIncrement
      }
   };
   static PStepCheckLimit s_stepCheckLimit;


   // processes the first result of the iteration (use all the parameters)
   class PStepCheckResult: public PStep {
   public:
      PStepCheckResult(){ apply = apply_;}
      virtual ~PStepCheckResult() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_ffor::PStepCheckResult"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         // Empty sequence? -- return nil
         if( ctx->topData().isBreak() )
         {
            ctx->returnFrame();
            return;
         }
         ctx->popCode();

         // check
         ctx->pushCode( &s_stepCheckLimit );

         // Well push twice, no problem with that.
         ctx->pushCode( static_cast<PStep*>(ctx->param(2)->asInst()) );
         ctx->pushCode( static_cast<PStep*>(ctx->param(1)->asInst()) );
      }
   };
   static PStepCheckResult s_stepAdvance;


   // This is invoked after op_iter is called.
   class PStepIncrement: public PStep {
   public:
      PStepIncrement(){ apply = apply_;}
      virtual ~PStepIncrement() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_ffor::PStepIncrement"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         ctx->popData();
         // step into checking the result
         ctx->pushCode( &s_stepAdvance );
         ctx->pushCode( static_cast<PStep*>(ctx->param(3)->asInst()) );
      }
   };
   static PStepIncrement s_stepIncrement;

   if( pCount < 4 )
   {
      throw paramError(__LINE__, SRC);
   }

   // prepare the code that will do the code
   ctx->pushCode( &s_stepIncrement );

   // this is an ETA functions, parameters are always expressions.
   ctx->pushCode( static_cast<PStep*>(ctx->param(0)->asInst()) );
}

/*#
  @function makeEnum
  @brief Creates an enumerated set of symbols.
  @param ... Names of the symbols to be enumerated.

  This funcion creates a set of N symbols in the caller context that are enumerated 0 to N-1.

  For example:
  @code
  makeEnum( "one", "two", "three" )
  > ~one    // 0
  > ~two    // 1
  > ~three  // 2
  @endcode

  @notee The '~' prepended to the symbol names in the example is used to
  prevent the module compiler to generate an implicit import request for
  symbols that are not found in the source code.

*/

FALCON_DEFINE_FUNCTION_P(makeEnum)
{
   // This is called after op_next generates the next item.
   class PStepGen: public PStep {
   public:
      PStepGen(){ apply = apply_;}
      virtual ~PStepGen() {}
      virtual void describeTo( String& tgt ) const { tgt = "Function_makeEnum::PStepGen"; }

      static void apply_(const PStep*, VMContext* ctx )
      {
         int pCount = ctx->currentCode().m_seqId;
         while( pCount > 0 )
         {
            // we should have checked before entering this pstep
            fassert( ctx->topData().isString() || ctx->topData().isSymbol() );
            const String* name = ctx->topData().isString() ? ctx->topData().asString() : &ctx->topData().asSymbol()->name();
            Item* value = ctx->resolveSymbol( *name, true );
            value->setInteger(pCount-1);
            --pCount;
            ctx->popData();
         }

         // clear the item that shall be seen as return value by the caller.
         ctx->topData().setNil();
         // out of business
         ctx->popCode();
      }
   };
   static PStepGen s_stepNext;

   // we have nothing to do without parameters.
   if( pCount == 0 )
   {
      ctx->returnFrame();
      return;
   }

   // check that all the parameters are strings
   for(int i = 0; i < pCount; ++i )
   {
      Item* i_param = ctx->param(i);
      if( ! (i_param->isString() || i_param->isSymbol()) )
      {
         throw paramError("*S|$");
      }
   }


   // we'll use a bit of a trick; we modify the base of the call stack so that
   // we can still access the parameters after we return the frame.
   ctx->pushData(Item()); // create a nil that can be changed
   ctx->currentFrame().m_dataBase += pCount+1;
   ctx->returnFrame();
   ctx->popData(); // here
   ctx->pushCode( &s_stepNext );
   ctx->currentCode().m_seqId = pCount;
   s_stepNext.apply(&s_stepNext, ctx);
}


}
}

/* end of sleep.cpp */
