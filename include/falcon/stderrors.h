/*
   FALCON - The Falcon Programming Language
   FILE: stderrors.h

   Engine static/global data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 15:30:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

// allow multiple inclusions
/*
 * \file Engine level standard error.
 *
 * This file is a C++ preprocessor file that declares and eventually defines
 * the list of standard errors knows by the falcon engine.
 *
 * It is built so that, if normally included, it simply gives a list of
 * Error and ClassError entities declared by the engine; in this mode, it can
 * be included multiple times as any other header file (the definition will be
 * expanded just once).
 *
 * The falcon engine (in engine/engine.cpp) uses this list to create a set
 * of Engine::addError and Engine::addMantra invocations, to that the errors
 * are known and available to any script directly as builtins. This is done by
 * prefixing a second inclusion with the directive
 *
 * @code
 *   #define FALCON_IMPLEMENT_ENGINE_ERRORS
 * @endcode
 *
 * directly in the body of the function that is required to create and account
 * for the given error entities.
 */
# ifndef FALCON_STDERRORS_H
#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/error.h>
#endif

#ifdef FALCON_IMPLEMENT_ENGINE_ERRORS
   #undef FALCON_DECLARE_ENGINE_ERROR
   #define FALCON_DECLARE_ENGINE_ERROR( __name__ )  \
       this->addMantra( this->registerError(new Class##__name__(false)) );
#else
   #ifdef FALCON_STDERRORS_H
      #undef FALCON_DECLARE_ENGINE_ERROR
      #define FALCON_DECLARE_ENGINE_ERROR( __name__ )
   #else
      #define FALCON_DECLARE_ENGINE_ERROR( __name__ )  FALCON_DECLARE_ERROR( __name__ )
   #endif
#endif

#define FALCON_STDERRORS_H

#ifndef FALCON_IMPLEMENT_ENGINE_ERRORS
namespace Falcon
{
#endif
   /** Class handler for AccessError exceptions.
    *
    * An AccessError is thrown when trying to access a property
    * that is not provided by an instance, or trying to access
    * an index that is out of range or not available.
    */
   FALCON_DECLARE_ENGINE_ERROR( AccessError )

   /** Class handler for AccessTypeError exceptions.
    *
    * This is thrown when the type of a data is incompatible
    * with the access that is trying to be performed. For instance,
    * trying to store a string value in a property that is accessing
    * numbers only will generate this error.
    *
    * Also, trying to access an index with a variable of an arbitrary
    * unallowed type might generate this error. For instance, this may
    * happen when trying to get a numeric index out of a string-hash table.
    */
   FALCON_DECLARE_ENGINE_ERROR( AccessTypeError )

   /** Class handler for CodeError exceptions.
    *
    * Code errors can be thought as "back to the drawing board" level errors.
    * They indicate an error condition that shouldn't happen in a well designed
    * program, and that require a fix in the code so that the condition doesn't
    * happen anymore.
    */
   FALCON_DECLARE_ENGINE_ERROR( CodeError )

   /** Class handler for Concurrency exceptions.
    *
    * Thrown when an impossible concurrency condition is detected in
    * parallel code, or when an object that was required to stay consistent
    * across concurrent operation was altered.
    */
   FALCON_DECLARE_ENGINE_ERROR( ConcurrencyError )



   /** Class handler for EncodingError exceptions.
    *
    * Generated when a transcoder detects an error
    * in the underlying stream.
    *
    */
   FALCON_DECLARE_ENGINE_ERROR( EncodingError )

   /** Class handler for GenericError exceptions.
    *
    * This errors are generated for mainly unknown errors.
    * Also, they are often used as simple collection of heterogeneous errors
    * that happened during a complex operation.
    */
   FALCON_DECLARE_ENGINE_ERROR( GenericError )


   /** Class handler for InterruptedError exceptions.
    * Deprecated.
    */
   FALCON_DECLARE_ENGINE_ERROR( InterruptedError )

   /** Class handler for IOError exceptions.
    *
    * Generated in case of VFS input or output errors, and/or underlying device
    * communication failure.
    */

   FALCON_DECLARE_ENGINE_ERROR( IOError )

   /** Class handler for LinkError exceptions.
    *
    * Generated when a static link request fails.
    *
    * Static link requests are declarations in modules asking the engine
    * to load a certain module or import a certain external variable on
    * their behalf.
    *
    */
   FALCON_DECLARE_ENGINE_ERROR( LinkError )


   /** Class handler for OperandError exceptions.
    *
    * This errors are generated when the type of operands of a certain
    * operation fails. For instance, it might be thrown while performing
    * "a + b" if the object "a" doesn't have the ability to add the b
    * object.
    *
    * In some cases, i.e. when the a object has addition overrides for
    * various types, but not for the one at stake, this might generate
    * a TypeError instead.
    */
   FALCON_DECLARE_ENGINE_ERROR( OperandError )

   /** Class handler for ParseError exceptions.
    *
    * This error is used historically by legacy Falcon source files.
    *
    * In some cases, i.e. when the a object has addition overrides for
    * various types, but not for the one at stake, this might generate
    * a TypeError instead.
    */
   FALCON_DECLARE_ENGINE_ERROR( ParseError )



   /** Class handler for MathError exceptions.
    *
    * Errors like division by zero or out of range calculus
    * are reported through this exception.
    */
   FALCON_DECLARE_ENGINE_ERROR( MathError )


   /** Class handler for ParamError exceptions.
    *
    * Generated while invoking evaluations (mainly when calling functions),
    * when the type of a certain parameter, or the
    * requirements for the parameters in general are not met.
    *
    */
   FALCON_DECLARE_ENGINE_ERROR( ParamError )


   /** Class handler for RangeError exceptions.
    *
    * Thrown when a value is out of range.
    *
    */
   FALCON_DECLARE_ENGINE_ERROR( RangeError )

   /** Class handler for SyntaxError exceptions.
    *
    * Generated by the compiler when it can't parse
    * source Falcon code. It might be also generated
    * by arbitrary code when trying to parse a user-defined
    * grammar, and failing to do so (for instance, when
    * dealing with a malformed INI file).
    *
    */
   FALCON_DECLARE_ENGINE_ERROR( SyntaxError )


   /** Class handler for TypeError exceptions.
    *
    * Generated in some occasions when the type (or class) of a variable
    * is not the expected one.
    *
    */
   FALCON_DECLARE_ENGINE_ERROR( TypeError )

   /** Class handler for Uncaught exceptions.
    *
    * This is thrown when an item, which is not an entity of a class derived from the Error class hierarchy,
    *  is explicitly raised through a "raise" statement, but isn't caught by any try/catch statement
    *  and reaches the top-level of the processor/context.
    *
    *  When this happens, the item is wrapped in an UncaughtError, and rethrown in this form.
    */
   FALCON_DECLARE_ENGINE_ERROR( UncaughtError )

   /** Class handler for UnserializableError exceptions.
    *
    * This is thrown when an object that is pushed in a Storer
    * is not supporting the serialization process, or when a
    * deserialization fails due to any problem not directly related
    * with a I/O issue.
    */
   FALCON_DECLARE_ENGINE_ERROR( UnserializableError )

   /** Class handler for UnsupportedError exceptions.
    *
    * Generated when invoking an operation that available, in general,
    * but not supported by the given instance. For instance, it's
    * generated when trying to write to a read-only stream.
    *
    */
   FALCON_DECLARE_ENGINE_ERROR( UnsupportedError )

#ifndef FALCON_IMPLEMENT_ENGINE_ERRORS
}
#endif

/* end of stderrors.h */
