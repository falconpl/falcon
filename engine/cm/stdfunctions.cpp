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

#include <falcon/cm/iterator.h>
#include <falcon/itemarray.h>

#include <falcon/error.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/matherror.h>
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

   @note As this call is performed, any critical section is abandoned, and aquired
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

/*#
   @function epoch
   @ingroup general_purpose
   @brief Returns the number of seconds since the "epoch" (1 Jan 1970).
   @return An integer number of seconds.
*/
FALCON_DEFINE_FUNCTION_P1(epoch)
{
   MESSAGE1( "-- called with 0 params" );
   int64 ep = Sys::_epoch();
   ctx->returnFrame(ep);
}

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

   This function terminates the current process as soon
   as possible.
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

   IteratorCarrier* ic;
   Class* iterClass=0;

   if( ctx->readInit().isNil() )
   {
      iterClass = module()->getClass("Iterator");
      fassert( iterClass != 0 );
      ic = new IteratorCarrier( *ctx->param(0) );

      ctx->writeInit(FALCON_GC_STORE(iterClass, ic));
   }
   else {
      const Item& initItem = ctx->readInit();
      void* data = 0;
      initItem.asClassInst( iterClass, data );
      fassert( iterClass == module()->getClass("Iterator") );
      ic = static_cast<IteratorCarrier*>(data);
   }

   // change into the class method, and use its return frames.
   ctx->param(0)->setBoolean(false);
   static_cast<ClassIterator*>(iterClass)->invokeDirectNextMethod(ctx, ic, pCount);
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

      for( int i = func->paramCount(); i < ctx->paramCount(); ++i )
      {
         retval->append( *ctx->param(i) );
      }
      ctx->topData() = FALCON_GC_HANDLE(retval);
   }

}


#if 0

/*#
   @function stdInRaw
   @brief Creates a stream that interfaces the standard input stream of the host process.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps input operations on the standard input of the
   process hosting the script. The returned stream is bound directly with the
   process input stream, without any automatic transcoding applied.
   @a Stream.readText will read the text as stream of binary data coming from the
   stream, unless @a Stream.setEncoding is explicitly called on the returned
   instance.

   Closing this stream has the effect to close the standard input stream of the
   process running the script (if the operation is allowed by the embedding
   application).  Applications trying to write data to the script process will be
   notified that the script has closed the stream and is not willing to receive
   data anymore.

   The stream is read only. Write operations will cause an I/O to be raised.
*/
FALCON_DEFINE_FUNCTION_P(stdInRaw)
{
   TRACE1( "stdInRaw -- called with %d params", pCount );
   (void) pCount;

   Stream* retStream = new StdInStream(false);
   ctx->returnFrame(FALCON_GC_HANDLE(retStream));
}

/*#
   @function stdOutRaw
   @brief Creates a stream that interfaces the standard output stream of the host process.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps output operations on the standard output stream of the
   process hosting the script. The returned stream is bound directly with the
   process output, without any automatic transcoding applied. @a Stream.writeText
   will write the text as stream of bytes to the stream, unless
   @a Stream.setEncoding is explicitly called on the returned instance.

   Closing this stream has the effect to close the standard output of the process
   running the script (if the operation is allowed by the embedding application).
   Print functions, fast print operations, default error reporting and so on will
   be unavailable from this point on.

   Applications reading from the output stream of the process running the scripts,
   in example, piped applications, will recognize that the script has completed
   its output, and will disconnect immediately, while the script may continue to run.

   The stream is write only. Read operations will cause an IoError to be raised.
*/

FALCON_FUNC  stdOutRaw ( ::Falcon::VMachine *vm )
{
   Stream* retStream = new StdOutStream(false);
   ctx->returnFrame(FALCON_GC_HANDLE(retStream));
}

/*#
   @function stdErrRaw
   @brief Creates a stream that interfaces the standard error stream of the host process.
   @ingroup core_syssupport
   @return A new valid @a Stream instance on success.

   The returned stream maps output operations on the standard error stream of the
   process hosting the script. The returned stream is bound directly with the
   process error stream, without any automatic transcoding applied.
   @a Stream.writeText will write the text as stream of bytes to the stream,
   unless @a Stream.setEncoding is explicitly called on the returned
   instance.

   Closing this stream has the effect to close the standard error stream of the
   process running the script (if the operation is allowed by the embedding
   application).  Applications reading from the error stream of the script will be
   notified that the stream has been closed, and won't be left pending in reading
   this stream.

   The stream is write only. Read operations will cause an I/O to be raised.
*/
FALCON_FUNC  stdErrRaw ( ::Falcon::VMachine *vm )
{
   Stream* retStream = new StdInStream(false);
   ctx->returnFrame(FALCON_GC_HANDLE(retStream));
}

/*# @endset */

/*#
   @function systemErrorDescription
   @ingroup general_purpose
   @brief Returns a system dependent message explaining an integer error code.
   @param errorCode A (possibly) numeric error code that some system function has returned.
   @return A system-specific error description.

   This function is meant to provide the users (and the developers) with a
   minimal support to get an hint on why some system function failed, without
   having to consult the system manual pages. The fsError field of the Error class
   can be fed directly inside this function.
*/

FALCON_FUNC  systemErrorDescription ( ::Falcon::VMachine *vm )
{
   Item *number = vm->param(0);
   if ( ! number->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) );
   }

   CoreString *str = new CoreString;
   ::Falcon::Sys::_describeError( number->forceInteger(), *str );
   vm->retval( str );
}

#endif

}
}

/* end of sleep.cpp */
