/*
   FALCON - The Falcon Programming Language.
   FILE: functional_ext.cpp

   Functional programming support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 02:10:57 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"

namespace Falcon {
namespace core {

/*#
   @funset functional_support Functional programming support
   @brief ETA functions and functional constructs.

   Falcon provides some special functional programming constructs that are known
   to the VM to have special significance. The vast majority of them starts a
   "functional evaluation" chain on their parameters before their value is evaluated.
   A functional evaluation is a recursive evaluation (reduction) of list structures into
   atoms. At the moment, the only list structure that can be evaluated this way is the array.
   Evaluating a parameter in functional context means that the given parameter will be
   recursively scanned for callable arrays or symbols that can be reduced to atoms. A callable
   array is reduced by calling the function and substituting it with its return value.
   When all the contents of the list are reduced, the higher level is evaluated.

   Consider this example:
   @code
   function func0( p0, p1 ): ...
   function func1( p0 ): ...

   list = [func0, [func1, param1], param2]
   @endcode

   Calling @b list as a callable array, func0 will be called with the array [func1, param1] as
   the first parameter, and param2 as the second parameter. On the other hand, evaluating
   the above list in a functional context, first func1 will be called with param1, then
   func0 will be called with the return value of the previous evaluation as the first parameter,
   and with param2 as the second parameter.

   The functions in this section are considered "special constructs" as the VM knows them and
   treats them specially. Their definition overrides the definition of a functional evaluation,
   so that when the VM finds a special construct in its evaluation process, it ceases using the
   default evaluation algorithm and passes evaluation control to the construct.

   For example, the iff construct selects one of its branches to be evaluated only if the first
   parameter evaluates to true:
   @code
   list = [iff, someValueIsTrue, [func0, [func1, param1]], [func1, param2] ]
   @endcode

   If this list had to be evaluated in a functional context, then before iff had a chance to
   decide what to do, the two arrays [func0, ...] and [func1,...] would have been evaluated.
   As iff is a special construct, the VM doesn't evaluate its parameters and lets iff perform
   its operations as it prefer. In the case o iff, it first evaluates the first parameter,
   then evaluates in functional context the second on the third parameter,
   leaving unevaluated the other one.

   Not all constructs evaluates everything it is passed to them in a functional context. Some of
   them are meant exactly to treat even a callable array (or anything else that should be reduced)
   as-is, stopping the evaluation process as the VM meets them. The description of each construct
   explains its working principles, and whether if its parameters are  evaluated or not.

   Please, notice that "callable" doesn't necessarily mean "evaluable". To evaluate in functional
   context a callable symbol without parameter, it must be transformed into a single-element array.
   For example:
   @code
   function func0(): ...

   result = [iff, shouldEval, [func0], func0]
   @endcode

   This places in result the value returned by func0 if shouldEval is true, while it returns exactly
   the function object func0 as-is if shouldEval is false.

   A more formal definition of the funcional programming support  in Falcon is provided in the
   Survival Guide.
*/

static bool core_any_next( ::Falcon::VMachine *vm )
{
   // was the elaboration succesful?
   if ( vm->regA().isTrue() )
   {
      vm->regA().setBoolean( true );
      return false;
   }

   // repeat checks.
   CoreArray *arr = vm->param(0)->asArray();
   uint32 count = (uint32) vm->local(0)->asInteger();
   while( count < arr->length() )
   {
      Item *itm = &arr->at(count);
      *vm->local(0) = (int64) count+1;
      if ( vm->functionalEval( *itm  ) )
      {
         return true;
      }
      else if ( vm->regA().isTrue() ) {
         vm->regA().setBoolean( true );
         return false;
      }
      count++;
   }

   vm->regA().setBoolean( false );
   return false;
}

/*#
   @function any
   @inset functional_support
   @brief Returns true if any of the items in a given collection evaluate to true.
   @param sequence A sequence of arbitrary items.
   @return true at least one item in the collection is true, false otherwise.

   Items in @b sequence are evaluated in functional context for truth value. This means that,
   if they are sigmas, they get sigma-reduced and their return value is evaluated,
   otheriwise they are evaluated directly.

   Truth value is determined using the standard Falcon truth
   check (nil is false, numerics are true if not zero, strings and collections are true if not
   empty, object and classes are always true).

   The check is short circuited. This means that elements are evaluated until
   an element considered to be true (or sigma-reduced to a true value) is found.

   If the collection is empty, this function returns false.
*/
FALCON_FUNC  core_any ( ::Falcon::VMachine *vm )
{
   Item *i_param = vm->param(0);
   if( i_param == 0 || !i_param->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "A" ) );
   }

   CoreArray *arr = i_param->asArray();
   uint32 count = arr->length();
   vm->returnHandler( &core_any_next );
   vm->addLocals(1);

   for( uint32 i = 0; i < count; i ++ )
   {
      Item *itm = &arr->at(i);
      *vm->local(0) = (int64) i+1;
      if ( vm->functionalEval( *itm  ) )
      {
         return;
      }
      else if ( vm->regA().isTrue() ) {
         vm->returnHandler( 0 );
         vm->regA().setBoolean( true );
         return;
      }
   }

   vm->returnHandler( 0 );
   vm->regA().setBoolean( false );
}


static bool core_all_next( ::Falcon::VMachine *vm )
{
   // was the elaboration succesful?
   if ( ! vm->regA().isTrue() )
   {
      vm->regA().setBoolean( false );
      return false;
   }

   // repeat checks.
   CoreArray *arr = vm->param(0)->asArray();
   uint32 count = (uint32) vm->local(0)->asInteger();
   while( count < arr->length() )
   {
      Item *itm = &arr->at(count);

      *vm->local(0) = (int64) count+1;
      if ( vm->functionalEval( *itm  ) )
      {
         return true;
      }
      else if ( ! vm->regA().isTrue() ) {
         vm->regA().setBoolean( false );
         return false;
      }
      count++;
   }

   vm->regA().setBoolean( true );
   return false;
}

/*#
   @function all
   @inset functional_support
   @brief Returns true if all the items in a given collection evaluate to true.
   @param sequence A sequence of arbitrary items.
   @return true if all the items are true, false otherwise

   Items in @b sequence are evaluated in functional context for truth value. This means that,
   if they are sigmas, they get sigma-reduced and their return value is evaluated,
   otheriwise they are evaluated directly.

   Truth value is determined using the standard Falcon truth
   check (nil is false, numerics are true if not zero, strings and collections are true if not
   empty, object and classes are always true).

   The check is short circuited. This means that the processing of parameters
   is interrupted as an element is evaluated into false.

   If the collection is empty, this function returns false.
*/

FALCON_FUNC  core_all ( ::Falcon::VMachine *vm )
{
   Item *i_param = vm->param(0);
   if( i_param == 0 || !i_param->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "A" ) );
   }

   CoreArray *arr = i_param->asArray();
   uint32 count = arr->length();
   if ( count == 0 )
   {
      vm->regA().setBoolean( false );
      return;
   }

   vm->returnHandler( &core_all_next );
   vm->addLocals(1);

   for( uint32 i = 0; i < count; i ++ )
   {
      Item *itm = &arr->at(i);
      *vm->local(0) = (int64) i+1;

      if ( vm->functionalEval( *itm  ) )
      {
         return;
      }
      else if ( ! vm->regA().isTrue() ) {
         vm->returnHandler( 0 );
         vm->regA().setBoolean( false );
         return;
      }
   }

   vm->returnHandler( 0 );
   vm->regA().setBoolean( true );
}


static bool core_anyp_next( ::Falcon::VMachine *vm )
{
   // was the elaboration succesful?
   if ( vm->regA().isTrue() )
   {
      vm->regA().setBoolean( true );
      return false;
   }

   // repeat checks.
   int32 count = (uint32) vm->local(0)->asInteger();
   while( count < vm->paramCount() )
   {
      Item *itm = vm->param( count );
      *vm->local(0) = (int64) count+1;

      if ( vm->functionalEval( *itm  ) )
      {
         return true;
      }
      else if ( vm->regA().isTrue() ) {
         vm->regA().setBoolean( true );
         return false;
      }
      count++;
   }

   vm->regA().setBoolean( false );
   return false;
}

/*#
   @function anyp
   @inset functional_support
   @brief Returns true if any one of the parameters evaluate to true.
   @param ... A list of arbitrary items.
   @return true at least one parameter is true, false otherwise.

   This function works like @a any, but the sequence may be specified directly
   in the parameters rather being given in a separate array. This make easier to write
   anyp in callable arrays. For example, one may write
   @code
      [anyp, 1, k, n ...]
   @endcode
   while using any one should write
   @code
      [any, [1, k, n ...]]
   @endcode

   Parameters are evaluated in functional context. This means that,
   if they are sigmas, they get sigma-reduced and their return value is evaluated,
   otheriwise they are evaluated directly.

   Truth value is determined using the standard Falcon truth
   check (nil is false, numerics are true if not zero, strings and collections are true if not
   empty, object and classes are always true).

   If called without parameters, this function returns false.
*/
FALCON_FUNC  core_anyp ( ::Falcon::VMachine *vm )
{
   uint32 count = vm->paramCount();
   vm->returnHandler( &core_anyp_next );
   vm->addLocals(1);

   for( uint32 i = 0; i < count; i ++ )
   {
      Item *itm = vm->param(i);
      *vm->local(0) = (int64) i+1;

      if ( vm->functionalEval( *itm  ) )
      {
         return;
      }
      else if ( vm->regA().isTrue() ) {
         vm->returnHandler( 0 );
         vm->regA().setBoolean( true );
         return;
      }
   }

   vm->returnHandler( 0 );
   vm->regA().setBoolean( false );
}


static bool core_allp_next( ::Falcon::VMachine *vm )
{
   // was the elaboration succesful?
   if ( ! vm->regA().isTrue() )
   {
      vm->regA().setBoolean( false );
      return false;
   }

   // repeat checks.
   int32 count = (uint32) vm->local(0)->asInteger();
   while( count < vm->paramCount() )
   {
      Item *itm = vm->param(count);

      *vm->local(0) = (int64) count+1;
      if ( vm->functionalEval( *itm  ) )
      {
         return true;
      }
      else if ( ! vm->regA().isTrue() ) {
         vm->regA().setBoolean( false );
         return false;
      }
      count++;
   }

   vm->regA().setBoolean( true );
   return false;
}

/*#
   @function allp
   @inset functional_support
   @brief Returns true if all the parameters evaluate to true.
   @param ... An arbitrary list of items.
   @return true if all the items are true, false otherwise

   This function works like @a all, but the collection may be specified directly
   in the parameters rather being given in a separate array. This make easier to
   write allp in callable arrays. For example, one may write
   @code
      [allp, 1, k, n ...]
   @endcode
   while using all one should write
   @code
      [all, [1, k, n ...]]
   @endcode

   Parameters are evaluated in functional context. This means that,
   if they are sigmas, they get sigma-reduced and their return value is evaluated,
   otheriwise they are evaluated directly.

   Truth value is determined using the standard Falcon truth
   check (nil is false, numerics are true if not zero, strings and collections are true if not
   empty, object and classes are always true).

   If called without parameters, this function returns false.
*/
FALCON_FUNC  core_allp ( ::Falcon::VMachine *vm )
{
   uint32 count = vm->paramCount();
   vm->returnHandler( &core_allp_next );
   vm->addLocals(1);

   if ( count == 0 )
   {
      vm->retval(0);
      return;
   }

   for( uint32 i = 0; i < count; i ++ )
   {
      Item *itm = vm->param(i);
      *vm->local(0) = (int64) i+1;
      if ( vm->functionalEval( *itm  ) )
      {
         return;
      }
      else if ( ! vm->regA().isTrue() ) {
         vm->returnHandler( 0 );
         vm->regA().setBoolean( false );
         return;
      }
   }

   vm->returnHandler( 0 );
   vm->retval( 1 );
}


/*#
   @function eval
   @inset functional_support
   @brief Evaluates a sequence in functional context.
   @param sequence A sequence to be evaluated.
   @return The sigma-reduction (evaluation) result.

   The parameter is evaluated in functional context; this means that if the parameter
   is a sequence starting with a callable item, that item gets called with the rest of the
   sequence passed as parameters, and the result it returns is considered the
   "evaluation result". This is performed recursively, inner-to-outer, on every element
   of the sequence before the call to the first element is actually performed.

   The description of the functional evaluation algorithm is included in the heading
   of this section.
*/

FALCON_FUNC  core_eval ( ::Falcon::VMachine *vm )
{
   Item *i_param = vm->param(0);
   if( i_param == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "X" ) );
   }
   uint32 pcount = vm->paramCount() - 1;
   for ( uint32 i = pcount; i > 0; i-- )
   {
      vm->pushParameter( *vm->param(i) );
   }

   vm->functionalEval( *i_param, pcount );
}

/*#
   @function valof
   @inset functional_support
   @brief Calls callable items or returns non callable ones.
   @param item The item to be checked.
   @return The item if it is not callable, or the call return value.

   The name function is a short for @i extended @i value. It is meant
   to determine if the passed item is a non-callable value or if it
   should be called to determine a value. Performing this check at
   script level time consuming and often clumsy, and this function
   is easily used in functional sequences.
*/

FALCON_FUNC  core_valof ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 )
   {
      vm->retnil();
      return;
   }

   Item *elem = vm->param( 0 );
   if( elem->isCallable() )
      vm->callFrame( *elem, 0 );
   else
      vm->retval( *elem );
}

/*#
   @function min
   @inset functional_support
   @brief Picks the minimal value among its parameters.
   @param ... The items to be checked.
   @return The smallest item in the sequence.

   This function performs a lexicographic minority check
   on each element passed as a parameter, returning the
   smallest of them.

   A standard VM comparation is performed, so the standard
   ordering rules apply. This also means that objects overloading
   the @a BOM.compare method may provide specialized ordering
   rules.

   If more than one item is found equal and lesser than
   all the others, the first one is returned.

   If the function is called without parameters, it returns @b nil.
*/

FALCON_FUNC  core_min ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 )
   {
      vm->retnil();
      return;
   }

   Item *elem = vm->param( 0 );
   for ( int32 i = 1; i < vm->paramCount(); i++)
   {
      if ( *vm->param(i) < *elem )
      {
         elem = vm->param(i);
      }
   }

   vm->retval( *elem );
}

/*#
   @function max
   @inset functional_support
   @brief Picks the maximum value among its parameters.
   @param ... The items to be checked.
   @return The greatest item in the sequence.

   This function performs a lexicographic majority check
   on each element passed as a parameter, returning the
   greater of them.

   A standard VM comparation is performed, so the standard
   ordering rules apply. This also means that objects overloading
   the @a BOM.compare method may provide specialized ordering
   rules.

   If more than one item is found equal and greater than
   all the others, the first one is returned.

   If the function is called without parameters, it returns @b nil.
*/

FALCON_FUNC  core_max ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 )
   {
      vm->retnil();
      return;
   }

   Item *elem = vm->param( 0 );
   int32 count = vm->paramCount();
   for ( int32 i = 1; i < count; i++)
   {
      if ( *vm->param(i) > *elem )
      {
         elem = vm->param(i);
      }
   }

   vm->retval( *elem );
}

/*#
   @function map
   @inset functional_support
   @brief Creates a new vector of items transforming each item in the original array through the mapping function.
   @param mfunc A function or sigma used to map the array.
   @param sequence A sequence of arbitrary items.
   @return The parameter unevaluated.

   mfunc is called iteratively for every item in the collection; its return value is added to the
   mapped array. In this way it is possible to apply an uniform transformation to all the item
   in a collection.

   If mfunc returns an out of band nil item, map skips the given position in the target array,
   actually acting also as a filter function.

   For example:
   @code
      function mapper( item )
         if item < 0: return oob(nil)  // discard negative items
         return item ** 0.5            // perform square root
      end

   inspect( map( mapper, [ 100, 4, -12, 9 ]) )    // returns [10, 2, 3]
   @endcode

   @see oob
*/

static bool core_map_next( ::Falcon::VMachine *vm )
{
   // callable in first item
   CoreArray *origin = vm->param(1)->asArray();
   uint32 count = (uint32) vm->local(0)->asInteger();
   CoreArray *mapped = vm->local(1)->asArray();

   if ( ! vm->regA().isOob() )
      mapped->append( vm->regA() );

   if ( count < origin->length() )
   {
      *vm->local(0) = (int64) count + 1;
      vm->pushParameter( origin->at(count) );
      vm->callFrame( *vm->param(0), 1 );
      return true;
   }

   vm->retval( mapped );
   return false;
}


FALCON_FUNC  core_map ( ::Falcon::VMachine *vm )
{
   Item *callable = vm->param(0);
   Item *i_origin = vm->param(1);
   if( callable == 0 || !callable->isCallable() ||
       i_origin == 0 || !i_origin->isArray()
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "C,A" ) );
   }

   CoreArray *origin = i_origin->asArray();
   CoreArray *mapped = new CoreArray( origin->length() );
   if ( origin->length() > 0 )
   {
      vm->returnHandler( &core_map_next );
      vm->addLocals( 2 );
      *vm->local(0) = (int64)1;
      *vm->local(1) = mapped;

      vm->pushParameter( origin->at(0) );
      // do not use pre-fetched pointer
      vm->callFrame( *vm->param(0), 1 );
      return;
   }

   vm->retval( mapped );
}


static bool core_dolist_next ( ::Falcon::VMachine *vm )
{
   CoreArray *origin = vm->param(1)->asArray();
   uint32 count = (uint32) vm->local(0)->asInteger();

   // done -- let A stay as is.
   if ( count >= origin->length() )
      return false;

   //if we called
   if ( vm->local(1)->asInteger() == 1 )
   {
      // not true? -- exit
      if ( vm->regA().isOob() && vm->regA().isNil() )
      {
         return false;
      }

      // prepare for next loop
      *vm->local(1) = (int64)0;
      if ( vm->functionalEval( origin->at(count) ) )
      {
         return true;
      }
   }

   *vm->local(0) = (int64) count + 1;
   *vm->local(1) = (int64) 1;
   vm->pushParameter( vm->regA() );
   vm->callFrame( *vm->param(0), 1 );
   return true;
}


/*#
   @function dolist
   @inset functional_support
   @brief Repeats an operation on a list of parameters.
   @param processor A callable item that will receive data coming from the sequence.
   @param sequence A list of items that will be fed in the processor one at a time.
   @optparam ... Optional parameters to be passed to the first callable item.
   @return The return value of the last callable item.

   Every item in @b sequence is passed as parameter to the processor, which must be a callable
   item. Items are also functionally evaluated, one by one, but the parameter @b sequence is not
   functionally evaluated as a whole; to do that, use the explicit evaluation:
   @code
      dolist( processor, eval(array) )
   @endcode
   This method is equivalent to @a xmap, but it has the advantage that it doesn't create an array
   of evaluated results. So, when it is not necessary to transform a sequence in another through a
   mapping function, but just to run repeatedly over a collection, this function is to be preferred.
*/
FALCON_FUNC  core_dolist ( ::Falcon::VMachine *vm )
{
   Item *callable = vm->param(0);
   Item *i_origin = vm->param(1);
   if( callable == 0 || !callable->isCallable() ||
       i_origin == 0 || !i_origin->isArray()
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "C,A" ) );
   }

   CoreArray *origin = i_origin->asArray();
   if ( origin->length() != 0 )
   {
      vm->returnHandler( &core_dolist_next );
      vm->addLocals( 2 );
      // count
      *vm->local(0) = (int64) 0;

      //exiting from an eval or from a call frame? -- 0 eval
      *vm->local(1) = (int64) 0;

      if ( vm->functionalEval( origin->at(0) ) )
      {
         return;
      }

      // count
      *vm->local(0) = (int64) 1;

      //exiting from an eval or from a call frame? -- 1 callframe
      *vm->local(1) = (int64) 1;
      vm->pushParameter( vm->regA() );
      vm->callFrame( *vm->param(0), 1 );
   }
}


static bool core_times_next_( ::Falcon::VMachine *vm )
{
   int64 end = vm->local(0)->asInteger();
   int64 step = vm->local(1)->asInteger();
   int64 start = vm->local(2)->asInteger();

   while ( (step > 0 && start >= end)
     || (step < 0 && start < end )
     || ( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
     )
   {
      vm->retval( start );
      return false;
   }

   vm->pushParameter( start );
   *vm->local(2) = start + step;
   Item *i_sequence = vm->local(3);
   if( i_sequence->isArray() )
   {
      vm->functionalEval( *i_sequence, 1, false );
   }
   else
   {
      vm->callFrame( *i_sequence, 1 );
   }

   return true;
}


/*#
   @function times
   @inset functional_support
   @brief Repeats a sequence a determined number of times.
   @param count Count of times to be repeated or non-open range.
   @param sequence A function or a Sigma sequence.
   @return Last index processed.

   This function is very similar to a functional for/in loop. It repeats a sequence
   of callable items in the @b sequence parameter a determined number of
   times. If the @b sequence parameter is a sequence, parametric evaluation is
   performed and the @b &1 late binding is filled with the value of the index; if
   it's a function, then it is called with the counter value added as the last parameter.

   If the evaluated parameter is a sequence, full deep sigma evaluation is performed
   at each loop.

   The loop index count will be given values from 0 to the required index-1 if @b count is numeric,
   or it will act as the for/in loop if @b count is a range.

   For example:

   @code

      function printParam( var )
         > "Parameter is... ", var
      end

      // the followings are equivalent
      times( 10, [printParam, &1] )
      times( 10, printParam )
   @endcode

   The following example prints a list of pair numbers between 2 and 10:

   @code
      times( [2:11:2],     // range 2 to 10+1, with step 2
         .[ printl "Index is now..." &1 ]
         )
   @endcode

   Exactly like @a floop, the flow of calls in @b times can be altered by the functions in sequence returning
   an out-of-band 0 or 1. If any function in the sequence returns an out-of-band 0, @b times terminates and
   return immediately (performing an operation similar to "break"). If a function returns an out-of-band 1,
   the rest of the items in @b sequence are ignored, and the loop is restarted with the index updated; this
   is equivalent to a functional "continue". For example:

   @code
   times( 10,
             // skip numbers less than 5
      .[ .[(function(x); if x < 5: return oob(1); end)  &1]
         .[printl &1]   // print the others
       ]
    )
   @endcode

   The @b times function return the last generated value for the index. A natural termination of @b times
   can be detected thanks to the fact that the index is equal to the upper bound of the range, while
   an anticipated termination causes @b times to return a different index. For example, if @b count is
   10, the generated index (possibly received by the items in @b sequence) will range from 0 to 9 included,
   and if the function terminates correctly, it will return 10. If a function in @b sequence returns an
   out-of-band 0, causing a premature termination of the loop, the value returned by times will be the loop
   index at which the out-of-band 0 was returned.

   @note Ranges [m:n] where m > n (down-ranges) terminate at n included; in that case, a succesful
   completion of @b times return one-past n.
*/

/*#
   @method times Integer
   @brief repeats a sequence a given number of times.
   @param sequence Function or sequence to be repeated.
   @return Last index processed.

   This method works exactly as the @b times function when
   the first parameter is a number.

   @see times
*/

/*#
   @method times Range
   @brief repeats a sequence a given number of times.
   @param sequence Function or sequence to be repeated.
   @return Last index processed.

   This method works exactly as the @b times function when
   the first parameter is a range.

   @see times
*/

FALCON_FUNC  core_times ( ::Falcon::VMachine *vm )
{
   Item *i_count;
   Item *i_sequence;

   if ( vm->self().isMethodic() )
   {
      i_count = &vm->self();
      i_sequence = vm->param(0);
   }
   else
   {
      i_count = vm->param(0);
      i_sequence = vm->param(1);
   }

   if( i_count == 0 || ! ( i_count->isRange() || i_count->isOrdinal() ) ||
       i_sequence == 0 || ! ( i_sequence->isArray() || i_sequence->isCallable() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "N|R, A|C" ) );
   }

   int64 start, end, step;
   if( i_count->isRange() )
   {
      if ( i_count->asRangeIsOpen() )
      {
         throw new ParamError( ErrorParam( e_inv_params )
            .origin(e_orig_runtime)
            .extra( "open range" ) );
      }

      start = i_count->asRangeStart();
      end = i_count->asRangeEnd();
      step = i_count->asRangeStep();
      if ( step == 0 ) step = start > end ? -1 : 1;
   }
   else {
      start = 0;
      end = i_count->forceInteger();
      step = end < 0 ? -1 : 1;
   }

   // check ranges and steps.
   if ( start == end ||
        ( start < end && ( step < 0 || start + step > end ) ) ||
        ( start > end && ( step > 0 || start + step < end ) )
    )
   {
      // no loop to be done.
      vm->retval( start );
      return;
   }

   // ok, we must do at least a loop
   vm->returnHandler( &core_times_next_ );

   // 0: shifting range
   // 1: position in the sequence calls.
   // 2: should evaluate ? 0 = no 1 = yes, 2 = already evaluating.
   vm->addLocals( 4 );
   // count
   *vm->local(0) = end;
   *vm->local(1) = step;
   *vm->local(2) = start;
   *vm->local(3) = *vm->param( vm->self().isMethodic()? 0 : 1 );

   // prevent dirty A to mess our break/continue system.
   vm->regA().setNil();

   // start the loop
   core_times_next_(vm);
}

/*#
      @method upto Integer
      @brief Repeats a function or a sequence until the upper limit is reached.
      @param llimit The upper limit of the loop.
      @param sequence The sequence or function to be repeated.
      @return The last index processed.

      This method repeats a loop from this integer down to the limit value included.
      If the limit is less than this integer, the function returns immediately.

      If the sequence is a function, then it is called iteratively with the current
      index value as last parameter. If it is a sequence, it is functionally evaluated
      and the &1 parametric binding is set to the index.
      @code
       2.upto( 5, printl )
       2.downto(5, .[printl "Value number " &1])
      @endcode

      In both cases, returning an oob(0) will cause the loop to terminate, while
      returning an oob(1) from any evaluation in the sequence makes the rest of
      the evaluation to be skipped and the loop to be restarted.
*/
FALCON_FUNC  core_upto ( ::Falcon::VMachine *vm )
{
   Item* i_count = vm->param(0);
   Item* i_sequence = vm->param(1);

   if( i_count == 0 || ! i_count->isOrdinal() ||
       i_sequence == 0 || ! ( i_sequence->isArray() || i_sequence->isCallable() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "N|R, A|C" ) );
   }

   int64 start =  vm->self().asInteger();
   int64 end = i_count->forceInteger();

   // check ranges and steps.
   if ( start > end )
   {
      // no loop to be done.
      vm->retval( start );
      return;
   }

   // ok, we must do at least a loop
   vm->returnHandler( &core_times_next_ );

   // 0: shifting range
   // 1: position in the sequence calls.
   // 2: should evaluate ? 0 = no 1 = yes, 2 = already evaluating.
   vm->addLocals( 4 );
   // count
   *vm->local(0) = end+1;
   *vm->local(1) = (int64) 1;
   *vm->local(2) = start;
   *vm->local(3) = *vm->param( 1 );

   // prevent dirty A to mess our break/continue system.
   vm->regA().setNil();

   // start the loop
   core_times_next_(vm);
}

/*#
      @method downto Integer
      @brief Repeats a function or a sequence until the lower limit is reached.
      @param llimit The lower limit of the loop.
      @param sequence The sequence or function to be repeated.
      @return The last index processed.

      This method repeats a loop from this integer down to the limit value included.
      If the limit is greater than this integer, the function returns immediately.

      If the sequence is a function, then it is called iteratively with the current
      index value as last parameter. If it is a sequence, it is functionally evaluated
      and the &1 parametric binding is set to the index.
      @code
       5.downto( 2, printl )
       3.downto(0, .[printl "Value number " &1])
      @endcode

      In both cases, returning an oob(0) will cause the loop to terminate, while
      returning an oob(1) from any evaluation in the sequence makes the rest of
      the evaluation to be skipped and the loop to be restarted.
*/
FALCON_FUNC  core_downto ( ::Falcon::VMachine *vm )
{
   Item* i_count = vm->param(0);
   Item* i_sequence = vm->param(1);

   if( i_count == 0 || ! i_count->isOrdinal() ||
       i_sequence == 0 || ! ( i_sequence->isArray() || i_sequence->isCallable() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "N|R, A|C" ) );
   }

   int64 start =  vm->self().asInteger();
   int64 end = i_count->forceInteger();

   // check ranges and steps.
   if ( start <= end )
   {
      // no loop to be done.
      vm->retval( start );
      return;
   }

   // ok, we must do at least a loop
   vm->returnHandler( &core_times_next_ );

   // 0: shifting range
   // 1: position in the sequence calls.
   // 2: should evaluate ? 0 = no 1 = yes, 2 = already evaluating.
   vm->addLocals( 4 );
   // count
   *vm->local(0) = end;
   *vm->local(1) = (int64) -1;
   *vm->local(2) = start;
   *vm->local(3) = *vm->param( 1 );

   // prevent dirty A to mess our break/continue system.
   vm->regA().setNil();

   // start the loop
   core_times_next_(vm);
}

static bool core_xmap_next( ::Falcon::VMachine *vm )
{
   // in vm->param(0) there is "callable".
   CoreArray *origin = vm->param(1)->asArray();
   uint32 count = (uint32) vm->local(0)->asInteger();
   CoreArray *mapped = vm->local(1)->asArray();


   if ( count < origin->length() )
   {
      if ( vm->local(2)->asInteger() == 1 )
      {
         if ( ! vm->regA().isOob() )
            mapped->append( vm->regA() );

         // prepare for next loop
         *vm->local(0) = (int64) count + 1;
         *vm->local(2) = (int64) 0;
         if ( vm->functionalEval( origin->at(count) ) )
         {
            return true;
         }
      }

      *vm->local(2) = (int64) 1;
      vm->pushParameter( vm->regA() );
      vm->callFrame( *vm->param(0), 1 );
      return true;
   }
   else {
      if ( ! vm->regA().isOob() )
            mapped->append( vm->regA() );
   }

   vm->retval( mapped );
   return false;
}


/*#
   @function xmap
   @inset functional_support
   @brief Creates a new vector of items transforming each item in the original array through the mapping function, applying also filtering on undesired items.
   @param mfunc A function or sigma used to map the array.
   @param sequence A sequence to be mapped.
   @return The mapped sequence.

   @b mfunc is called iteratively for every item in the collection;  its return value is added to
   the mapped array. Moreover, each item in the collection is functionally evaluated before
   being passed to mfunc.

   The filter function may return an out of band nil item to signal that the current item should
   not be added to the final collection.

    For example:
   @code

      mapper = { item => item < 0 ? oob(nil) : item ** 0.5 }
      add = { a, b => a+b }         // a block that will be evaluated

      inspect( xmap( mapper, [ [add, 99, 1], 4, -12, 9 ]) )    // returns [10, 2, 3]
   @endcode

   @see oob
   @see dolist
*/

FALCON_FUNC  core_xmap ( ::Falcon::VMachine *vm )
{
   Item *callable = vm->param(0);
   Item *i_origin = vm->param(1);
   if( callable == 0 || !callable->isCallable() ||
       i_origin == 0 || !i_origin->isArray()
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "C,A" ) );
   }

   CoreArray *origin = i_origin->asArray();
   CoreArray *mapped = new CoreArray( origin->length() );
   if ( origin->length() > 0 )
   {
      vm->returnHandler( &core_xmap_next );
      vm->addLocals( 3 );
      *vm->local(0) = (int64)1;
      *vm->local(1) = mapped;
      *vm->local(2) = (int64) 0;

      if ( vm->functionalEval( origin->at(0) ) )
      {
         return;
      }

      *vm->local(2) = (int64) 1;
      vm->pushParameter( vm->regA() );
      vm->callFrame( *vm->param(0), 1 );
      return;
   }

   vm->retval( mapped );
}

static bool core_filter_next ( ::Falcon::VMachine *vm )
{
   CoreArray *origin = vm->param(1)->asArray();
   CoreArray *mapped = vm->local(0)->asArray();
   uint32 count = (uint32) vm->local(1)->asInteger();

   if ( vm->regA().isTrue() )
      mapped->append( origin->at(count -1) );

   if( count == origin->length()  )
   {
      vm->retval( mapped );
      return false;
   }

   *vm->local(1) = (int64) count+1;
   vm->pushParameter( origin->at(count) );
   vm->callFrame( *vm->param(0), 1 );
   return true;
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

FALCON_FUNC  core_filter ( ::Falcon::VMachine *vm )
{
   Item *callable = vm->param(0);
   Item *i_origin = vm->param(1);
   if( callable == 0 || !callable->isCallable() ||
      i_origin == 0 || !i_origin->isArray()
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "C,A" ) );
   }

   CoreArray *origin = i_origin->asArray();
   CoreArray *mapped = new CoreArray( origin->length() / 2 );
   if( origin->length() > 0 )
   {
      vm->returnHandler( &core_filter_next );
      vm->addLocals(2);
      *vm->local(0) = mapped;
      *vm->local(1) = (int64) 1;
      vm->pushParameter( origin->at(0) );
      vm->callFrame( *vm->param(0), 1 );
      return;
   }

   vm->retval( mapped );
}


static bool core_reduce_next ( ::Falcon::VMachine *vm )
{
   // Callable in param 0
   CoreArray *origin = vm->param(1)->asArray();

   // if we had enough calls, return (the return value of the last call frame is
   // already what we want to return).
   uint32 count = (uint32) vm->local(0)->asInteger();
   if( count >= origin->length() )
      return false;

   // increment count for next call
   vm->local(0)->setInteger( count + 1 );

   // call next item
   vm->pushParameter( vm->regA() ); // last returned value
   vm->pushParameter( origin->at(count) ); // next element
   vm->callFrame( *vm->param(0), 2 );
   return true;
}

/*#
   @function reduce
   @inset functional_support
   @brief Uses the values in a given sequence and iteratively calls a reductor function to extract a single result.
   @param reductor A function or Sigma to reduce the array.
   @param sequence A sequence of arbitrary items.
   @optparam initial_value Optional startup value for the reduction.
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
   > reduce( {a,b=> a+b}, [], 0 )          // never calls lambda, returns 0
   > reduce( {a,b=> a+b}, [] )             // never calls lambda, returns Nil
   @endcode

   Items in the collection are treated literally (not evaluated).
*/
FALCON_FUNC  core_reduce ( ::Falcon::VMachine *vm )
{
   Item *callable = vm->param(0);
   Item *i_origin = vm->param(1);
   Item *init = vm->param(2);
   if( callable == 0 || !callable->isCallable()||
      i_origin == 0 || !i_origin->isArray()
      )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "C,A,[X]" ) );
   }

   CoreArray *origin = i_origin->asArray();
   vm->addLocals(1);
   // local 0: array position

   if ( init != 0 )
   {
      if( origin->length() == 0 )
      {
         vm->retval( *init );
         return;
      }

      vm->returnHandler( &core_reduce_next );
      vm->pushParameter( *init );
      vm->pushParameter( origin->at(0) );
      *vm->local(0) = (int64) 1;

      //WARNING: never use pre-cached item pointers after stack changes.
      vm->callFrame( *vm->param(0), 2 );
      return;
   }

   // if init == 0; if there is only one element in the array, return it.
   if ( origin->length() == 0 )
      vm->retnil();
   else if ( origin->length() == 1 )
      vm->retval( origin->at(0) );
   else
   {
      vm->returnHandler( core_reduce_next );
      *vm->local(0) = (int64) 2; // we'll start from 2

      // the first call is between the first and the second elements in the array.
      vm->pushParameter( origin->at(0) );
      vm->pushParameter( origin->at(1) );

      //WARNING: never use pre-cached item pointers after stack changes.
      vm->callFrame( *vm->param(0), 2 );
   }
}


static bool core_iff_next( ::Falcon::VMachine *vm )
{
   // anyhow, we don't want to be called anymore
   vm->returnHandler( 0 );

   if ( vm->regA().isTrue() )
   {
      if ( vm->functionalEval( *vm->param(1) ) )
         return true;
   }
   else
   {
      Item *i_ifFalse = vm->param(2);
      if ( i_ifFalse != 0 )
      {
         if ( vm->functionalEval( *i_ifFalse ) )
            return true;
      }
      else
         vm->retnil();
   }

   return false;
}


/*#
   @function iff
   @inset functional_support
   @brief Performs a functional if; if the first parameter evaluates to true, the second parameter is evaluated and then returned, else the third one is evaluated and returned.
   @param cfr A condition or a callable item.
   @param whenTrue Value to be called and/or returned in case cfr evaluates to true.
   @optparam whenFalse Value to be called and/or returned in case cfr evaluates to false.
   @return The evaluation result of one of the two branches (or nil).

   Basically, this function is meant to return the second parameter or the third (or nil if not given),
   depending on the value of the first parameter; however, every item is evaluated in a functional
   context. This means that cfr may be a callable item, in which case its return value will be evaluated
   for truthfulness, and also the other parameters may. For example:
   @code
      > iff( 0, "was true", "was false" )           // will print "was false"
      iff( [{a=>a*2}, 1] , [printl, "ok!"] )       // will print "ok!" and return nil
   @endcode

   In the last example, we are not interested in the return value (printl returns nil), but in executing
   that item only in case the first item is true. The first item is a callable item too, so iff will first
   execute the given block, finding a result of 2 (true), and then will decide which element to pick, and
   eventually execute. Notice that:
   @code
      iff( 1 , printl( "ok!" ), printl( "no" ) )
   @endcode

   This would have forced Falcon to execute the two printl calls before entering the iff function;
   still, iff would have returned printl return values (which is nil in both cases).
*/
FALCON_FUNC  core_iff ( ::Falcon::VMachine *vm )
{
   Item *i_cond = vm->param(0);
   Item *i_ifTrue = vm->param(1);
   Item *i_ifFalse = vm->param(2);

   if( i_cond == 0 || i_ifTrue == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "X,X,[X]" ) );
   }

   // we can use pre-fetched values as we have stack unchanged on
   // paths where we use item pointers.

   vm->returnHandler( &core_iff_next );
   if ( vm->functionalEval( *i_cond ) )
   {
      return;
   }
   vm->returnHandler( 0 );

   if ( vm->regA().isTrue() )
   {
      vm->functionalEval( *i_ifTrue );
   }
   else {
      if ( i_ifFalse != 0 )
         vm->functionalEval( *i_ifFalse );
      else
         vm->retnil();
   }
}


static bool core_choice_next( ::Falcon::VMachine *vm )
{
   if ( vm->regA().isTrue() )
   {
      vm->retval( *vm->param(1) );
   }
   else {
      Item *i_ifFalse = vm->param(2);
      if ( i_ifFalse != 0 )
         vm->retval( *i_ifFalse );
      else
         vm->retnil();
   }

   return false;
}

/*#
   @function choice
   @inset functional_support
   @brief Selects one of two alternatives depending on the evaluation of the first parameter.
   @param selector The item to be evaluated.
   @param whenTrue The item to return if selector evaluates to true.
   @optparam whenFalse The item to be returned if selector evaluates to false
   @optparam ... Optional parameters to be passed to the first callable item.
   @return The return value of the last callable item.

   The selector parameter is evaluated in functional context. If it's a true atom or if it's a
   callable array which returns a true value, the ifTrue parameter is returned as-is, else the
   ifFalse parameter is returned. If the ifFalse parameter is not given and the selector evaluates
   to false, nil is returned.

   The choice function is equivalent to iff where each branch is passed through the @a lit function:
   @code
      choice( selector, a, b ) == iff( selector, [lit, a], [lit, b] )
   @endcode
   In case a literal value is needed, choice is more efficient than using iff and applying lit on
   the parameters.
*/
FALCON_FUNC  core_choice ( ::Falcon::VMachine *vm )
{
   Item *i_cond = vm->param(0);
   Item *i_ifTrue = vm->param(1);
   Item *i_ifFalse = vm->param(2);

   if( i_cond == 0 || i_ifTrue == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
         extra( "X,X,[X]" ) );
   }

   vm->returnHandler( &core_choice_next );
   if ( vm->functionalEval( *i_cond ) )
   {
      return;
   }
   vm->returnHandler( 0 );

   if ( vm->regA().isTrue() )
   {
      vm->retval( *i_ifTrue );
   }
   else {
      if ( i_ifFalse != 0 )
         vm->retval( *i_ifFalse );
      else
         vm->retnil();
   }
}

/*#
   @function lit
   @inset functional_support
   @brief Return its parameter as-is
   @param item A condition or a callable item.
   @return The parameter unevaluated.

   This function is meant to interrupt functional evaluation of lists. It has
   the same meaning of the single quote literal ' operator of the LISP language.

   For example, the following code will return either a callable instance of printl,
   which prints a "prompt" before the parameter, or a callable instance of inspect:
   @code
      iff( a > 0, [lit, [printl, "val: "] ], inspect)( param )
   @endcode
   as inspect is a callable token, but not an evaluable one, it is already returned literally;
   however, [printl, "val:"] would be considered an evaluable item. To take its literal
   value and prevent evaluation in functional context, the lit construct must be used.
*/

FALCON_FUNC  core_lit ( ::Falcon::VMachine *vm )
{
   Item *i_cond = vm->param(0);

   if( i_cond == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "X" ) );
   }

   vm->regA() = *i_cond;
   // result already in A.
}


static bool core_cascade_next ( ::Falcon::VMachine *vm )
{
   // Param 0: callables array
   // local 0: counter (position)
   // local 1: last accepted result
   CoreArray *callables = vm->param(0)->asArray();
   uint32 count = (uint32) vm->local(0)->asInteger();

   // Done?
   if ( count >= callables->length() )
   {
      // if the last result is not accepted, return last accepted
      if ( vm->regA().isOob() )
      {
         // reset OOB, that may be set on first unaccepted parameter.
         vm->local(1)->resetOob();
         vm->retval( *vm->local(1) );
      }
      // else, just keep
      return false;
   }

   uint32 pc;

   // still some loop to do
   // accept result?
   if ( vm->regA().isOob() )
   {
      // not accepted.

      // has at least one parameter been accepted?
      if ( vm->local(1)->isOob() )
      {
         // no? -- replay initial params
         pc = vm->paramCount();
         for ( uint32 pi = 1; pi < pc; pi++ )
         {
            vm->pushParameter( *vm->param(pi) );
         }
         pc--;  //first param is our callable
      }
      else {
         // yes? -- reuse last accepted parameter
         pc = 1;
         vm->pushParameter( *vm->local(1) );
      }
   }
   else {
      *vm->local(1) = vm->regA();
      pc = 1;
      vm->pushParameter( vm->regA() );
   }

   // prepare next call
   vm->local(0)->setInteger( count + 1 );

   // perform call
   vm->callFrame( callables->at(count), pc ); // will throw noncallable in case of noncallable item.
   //throw new ParamError( ErrorParam( e_non_callable ).origin(e_orig_runtime) );


   return true;
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
FALCON_FUNC  core_cascade ( ::Falcon::VMachine *vm )
{
   Item *i_callables = vm->param(0);

   if( i_callables == 0 || !i_callables->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "A,..." ) );
   }

   // for the first callable...
   CoreArray *callables = i_callables->asArray();
   if( callables->length() == 0 )
   {
      vm->retnil();
      return;
   }

   // we have at least one callable.
   // Prepare the local space
   // 0: array counter
   // 1: saved previous value
   // saved previous value is initialized to oob until
   // someone accepts the first parameters.
   vm->addLocals(2);
   vm->local(0)->setInteger( 1 );  // we'll start from 1
   vm->local(1)->setOob();

   // echo the parameters to the first call
   uint32 pcount = vm->paramCount();
   for ( uint32 pi = 1; pi < pcount; pi++ )
   {
      vm->pushParameter( *vm->param(pi) );
   }
   pcount--;

   // install the handler
   vm->returnHandler( &core_cascade_next );

   // perform the real call
   vm->callFrame( callables->at(0), pcount );
   //throw new ParamError( ErrorParam( e_non_callable ).origin(e_orig_runtime) );
}


static bool core_floop_next ( ::Falcon::VMachine *vm )
{
   // Param 0: callables array
   CoreArray *callables = vm->param(0)->asArray();
   // local 0: counter (position)
   uint32 count = (uint32) vm->local(0)->asInteger();

   // next item.
   ++count;

   // still some loop to do
   if ( vm->regA().isInteger() && vm->regA().isOob() )
   {
      if ( vm->regA().asInteger() == 0 )
      {
         // we're done.
         vm->returnHandler( 0 ); // ensure we're not called after first loop
         vm->retnil();
         return false;
      }
      else if ( vm->regA().asInteger() == 1 )
      {
         // continue
         count = 0;
      }
   }

   if ( count >= callables->length() )
   {
      count = 0;
   }

   // save the count
   *vm->local(0) = (int64) count;
   // find a callable in the array
   if ( (*callables)[count].isCallable() )
   {
       vm->callFrame( (*callables)[count], 0 );
   }
   else
   {
      // set the item as A and recall ourself for evaluation
      vm->regA() = (*callables)[count];
      vm->recallFrame();
      return true;
   }

   // else, just return true
   return true;
}


/*#
   @function floop
   @inset functional_support
   @brief Repeats indefinitely a list of operations.
   @param sequence A sequence of callable items that gets called one after another.

   Every item in @b sequence gets executed, one after another. When the last element is executed,
   the first one is called again, looping indefinitely.
   Any function in the sequence may interrupt the loop by returning an out-of-band 0;
   if a function returns an out of band 1, all the remaining items in the list are ignored
   and the loop starts again from the first item.

   Items in the array are not functionally evaluated.
*/

FALCON_FUNC  core_floop ( ::Falcon::VMachine *vm )
{
   Item *i_callables = vm->param(0);

   if( i_callables == 0 || !i_callables->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "A" ) );
      return;
   }

   // for the first callable...
   CoreArray *callables = i_callables->asArray();
   if( callables->length() == 0 )
   {
      return;
   }

   // we have at least one callable.
   // Prepare the local space
   // 0: array counter
   vm->addLocals(1);
   vm->local(0)->setInteger( callables->length() );  // we'll start from 0 from the first loop

   // install the handler
   vm->returnHandler( &core_floop_next );

   // call it directly
   vm->regA().setNil(); // zero to avoid false signals to next handler
   vm->callFrameNow( &core_floop_next );
}

/*#
   @function firstOf
   @inset functional_support
   @brief Returns the first non-false of its parameters.
   @param ... Any number of arbitrary parameters.
   @return The first non-false item.

   This function scans the paraters one at a time. Sigma evaluation is stopped,
   or in other words, every parameters is considered as-is, as if @a lit was used on each of them.
   The function returns the first parameter being non-false in a standard Falcon truth check.
   Nonzero numeric values, non empty strings, arrays and dictionaries and any object is considered true.

   If none of the parameters is true, of is none of the parameter is given, the function returns nil
   (which is considered  false).
*/
FALCON_FUNC  core_firstof ( ::Falcon::VMachine *vm )
{
   int count = 0;
   Item *i_elem = vm->param(0);
   while( i_elem != 0 )
   {
      if ( i_elem->isTrue() )
      {
         vm->retval( *i_elem );
         return;
      }
      i_elem = vm->param( ++count );
   }

   vm->retnil();
}

/*#
   @function lbind
   @inset functional_support
   @brief Creates a dynamic late binding symbol.
   @param name A string representing a late binding name.
   @optparam value A future binding value.
   @return A newly created late binding name.

   This function create a late binding item which can be used
   in functional sequences as if the parameter was written in
   the source code prefixed with the amper '&' operator.

   The following lines are equivalent:
   @code
      bound = lbind( "count" )
      ctx = &count
   @endcode

   The return value of this function, both used directly or pre-cached,
   can be seamlessly merged with the & operator in functional sequences.

   For example, it is possible to write the following loop:
   @code
      eval( .[
         .[ times 10 &count .[
            .[eval .[ printl 'Counting...' .[lbind 'count'] ] ]
            ]
         ]] )
   @endcode

   It is also possible cache the value and use it afterwards:
   @code
      x = lbind( 'count' )
      eval( .[
         .[ times 10 &count .[
            .[ printl 'Counting...' x]
            ]
         ]] )
   @endcode

   The @b value parameter initializes a future (forward) binding.
   Future bindings are bindings with a potential value, which is applied
   during function calls and symbol resolution to pre-existing symbolic
   entities. In practice, they allow calling fucntions with named parameters.

   When mixing forward bindings and normal parameters, forward bindings are as
   placed directly at the position of the parameter they refer to, and they
   doesn't count during parameter expansion of non-named parameters. Also,
   they always overwrite the positional parameters, as they are considered
   after all the positional parameters have been placed on their spots.

   @code
      function test( par1, par2, par3 )
         >> "Par1: ", par1
         >> ", Par2: ", par2
         >  ", Par3: ", par3
      end

      x = lbind( "par2", "Hello" )

      test( x )                       // nil->par1, "Hello"->par2, nil->par3
      test( x, "Yo!" )                // "Yo!"->par1, "Hello"->par2, nil->par3
      test( "Yo!", x )                // as above
      test( "Yo!", "Yo! again", x )   // "Hello" overwrites "Yo again"
      test( x, "Yo!", "Yo! again", "end" )   // "Yo!"->par1, "Hello"->par2, "end"->par3
   @endcode

   @note lbind is @b not an ETA function.
*/
FALCON_FUNC  core_lbind ( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param(0);
   Item *i_value = vm->param(1);

   if( i_name == 0 || !i_name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "S" ) );
   }
   GarbageItem *itm = i_value == 0 ? 0 : new GarbageItem( *i_value );

   vm->regA().setLBind( new CoreString( *i_name->asString() ), itm );
}


static bool core_let_next( ::Falcon::VMachine *vm )
{
   *vm->param(0) = vm->regA();
   return false;
}


/*#
   @function let
   @inset functional_support
   @brief Assigns a value to another in a functional context.
   @param dest Destination value (passed by reference).
   @param source Source value.
   @return The assigned (source) value.

   This function assigns a literal value given in @b source into @b dest,
   provided dest is a late binding or is passed by referece.

   This function is an ETA and prevents evaluation of its first parameter. In other
   words, the first parameter is treadted as if passed through @a lit.
*/

FALCON_FUNC  core_let ( ::Falcon::VMachine *vm )
{
   Item *i_dest = vm->param(0);
   Item *i_source = vm->param(1);

   if( i_dest == 0 || ! vm->isParamByRef( 0 ) || i_source == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params )
         .origin(e_orig_runtime)
         .extra( "$X,X" ) );
   }

   vm->returnHandler( &core_let_next );
   if ( vm->functionalEval( *i_source ) )
   {
      return;
   }
   vm->returnHandler( 0 );

   *vm->param(0) = *vm->param(1);
   vm->regA() = *vm->param(0);
}


static bool core_brigade_next( ::Falcon::VMachine *vm )
{
   int64 next = vm->local(0)->asInteger();

   // Has the previous call returned something interesting?
   if ( vm->regA().isOob() )
   {
      if( vm->regA().isInteger() )
      {
         // break request?
         if( vm->regA().asInteger() == 0 )
         {
            vm->retnil();
            return false;
         }
         else if( vm->regA().asInteger() == 1 )
         {
            // loop from start
            next = 0;
         }
      }
      else if ( vm->regA().isArray() )
      {
         CoreArray* newParams = vm->regA().asArray();
         *vm->local(1) = newParams;
         // add a space for the calls
         newParams->prepend( Item() );
      }
   }

   CoreArray* list = vm->param(0)->asArray();

   // are we done?
   if( next >= list->length() )
      return false;

   // prepare the local call
   vm->local(0)->setInteger( next + 1 );

   // anyhow, prepare the call
   //-- have we changed parameters?
   if ( vm->local(1)->isArray() )
   {
      CoreArray* callarr = vm->local(1)->asArray();
      callarr->at(0) = list->at((int32)next);
      vm->callFrame( callarr, 0 );
   }
   else
   {
      // no? -- use our original paramters.
      for( int32 i = 1; i < vm->paramCount(); ++i )
      {
         vm->pushParameter( *vm->param(i) );
      }

      vm->callFrame( list->at((int32)next), vm->paramCount()-1 );
   }

   return true; // call me again
}


/*#
   @function brigade
   @inset functional_support
   @brief Process a list of functions passing the same parameters to them.
   @param fl The sequence of callable items to be called.
   @param ... Arbitrary parameters used by the brigade functions.
   @return The return value of the last function in fl.

   This function process a sequence of functions passing them the
   same set of parameters. The idea is that of a "brigate" of functions
   operating all on the same parameters so that it is possible to put
   common code into separate functions.

   Items in the list are not functionally evaluated; they are simply called,
   passing to them the same parameters that the brigade group receives. Brigate
   is an ETA funcion, and this means that ongoing functional evaluation is
   interrupted as soon as a brigade is encountered.

   @code
   function mean( array )
      value = 0
      for elem in array: value += elem
      return value / len( array )
   end

   function dbl( array )
      for i in [0:len(array)]: array[i] *= 2
   end

   doubleMean = .[ brigade .[
      dbl
      mean
   ]]

   > "Mean: ", mean( [1,2,3] )
   > "Double mean: ", doubleMean( [1,2,3] )
   @endcode

   The above example brigades a "prefix" function to double the values in
   an array that must be processed by the main function.

   Using out of band return values, the functions in the sequence can also
   control the parameters that the following functions will receive, terminate
   immediately the evaluation or restart it, forming a sort of iterative
   functional loop. An oob 0 causes the sequence to be interrutped (and return oob(0) itself),
   an out of band 1 will cause the first element of the sequence to be called again, and
   an out of band array will permanently change the parameters as seen by the functions.

   The following brigate performs a first step using the given parameters, and another one
   using modified ones:

   @code
   looper = .[brigade .[
      { val, text => printl( text, ": ", val ) } // do things
      { val, text => oob( [val+1, "Changed"] ) }  // change params
      { val, text => val >= 10 ? oob(0) : oob(1)}  // loop control
   ]]

   looper( 1, "Original" )
   @endcode
*/

FALCON_FUNC  core_brigade ( ::Falcon::VMachine *vm )
{
   Item *i_fl = vm->param(0);

   if( i_fl == 0 || ! i_fl->isArray() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin(e_orig_runtime)
         .extra( "A" ) );
   }

   // nothing to do?
   if ( i_fl->asArray()->length() == 0 )
   {
      vm->retnil();
      return;
   }

   vm->returnHandler( &core_brigade_next );
   vm->addLocals(2);
   vm->local(0)->setInteger(1);
   vm->local(1)->setNil();

   // anyhow, prepare the call
   for( int32 i = 1; i < vm->paramCount(); ++i )
   {
      vm->pushParameter( *vm->param(i) );
   }

   vm->callFrame( vm->param(0)->asArray()->at(0), vm->paramCount()-1 );
   //throw new ParamError( ErrorParam( e_non_callable,__LINE__ ).origin(e_orig_runtime) );
}


}
}

/* end of functional_ext.cpp */
