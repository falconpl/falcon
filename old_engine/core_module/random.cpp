/*
   FALCON - The Falcon Programming Language.
   FILE: random.cpp

   Random number related functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun nov 8 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

/** \file
   Random number generator related functions.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/vm.h>
#include <falcon/carray.h>
#include <falcon/string.h>
#include <falcon/sys.h>


#include <stdlib.h>
#include <math.h>

/*#
   @funset core_random Random functions
   @brief Functions providing random numbers and sequences.
   @beginset core_random
*/

namespace Falcon {
namespace core {

    class RandomCarrier : public FalconData
    {
    public:
        RandomCarrier() {};
        inline MTRand &getRNG(void) { return mtrand; }
        virtual RandomCarrier *clone() const { return NULL; } // not cloneable
        virtual void gcMark( uint32 mark ) {}
        virtual bool serialize( Stream *stream, bool bLive ) const { return false; }
        virtual bool deserialize( Stream *stream, bool bLive ) { return false; }
    private:
        MTRand mtrand;
    };

    FALCON_FUNC flc_Random_init( ::Falcon::VMachine *vm )
    {
        RandomCarrier *rc = new RandomCarrier();
        if ( vm->paramCount() )
        {
            // throw to indicate there is no other type yet we can use to seed the RNG
            rc->getRNG().seed( (uint32)vm->param(0)->forceIntegerEx() );
        }
        vm->self().asObject()->setUserData(rc);
    }

/*#
   @function random
   @brief Returns a pseudo random number.
   @param ... See below.
   @return A pseudo random number or a random item picked from parameters.

   This function has actually several functionalities that are
   selected depending on the parameters.

   Without parameters, the function returns a floating point number in the
   range [0,1).

   With a signle numeric parameter, the function returns an integer between
   0 and the number, included. The following functions are equivalent:
   @code
      > random( x )
      > int( random() * (x + 1) )
   @endcode

   With two numeric parameters, the function returns an integer in the range
   [x, y]. The following functions are equivalent:
   @code
      > random( x, y )
      > int( x + (random() * (y + 1)) )
   @endcode

   With more than two parameters, or when at least one of the first two
   parameters it not a number, one of the parameter is picked at random
   and then returned.

   The function @a randomChoice returns unambiguously one of the parameters
   picked at random.

*/
FALCON_FUNC  flc_random ( ::Falcon::VMachine *vm )
{
   int32 pcount = vm->paramCount();
   Item *elem1, *elem2;
   CoreObject *selfobj = vm->self().isNil() ? NULL : vm->self().asObject();
   MTRand &rng = selfobj ? ((RandomCarrier*)selfobj->getUserData())->getRNG() : vm->getRNG(); 

   switch( pcount )
   {
      case 0:
         vm->retval( (numeric) rng.rand() );
      break;

      case 1:
         elem1 = vm->param(0);
         if ( elem1->isOrdinal() ) {
            int64 num = elem1->forceInteger();
            if ( num < 0 )
               vm->retval( -((int64) rng.randInt64( (-num) & ~(UI64LIT(1) << 63) )) ); // mask out sign bit and make result negative
            else if ( num == 0 )
               vm->retval( 0 );
            else
               vm->retval( (int64) rng.randInt64( num & ~(UI64LIT(1) << 63) ) ); // mask out sign bit, result always positive
         }
         else
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime ).extra( "[N],[N]" ) );
      break;

      case 2:
         elem1 = vm->param(0);
         elem2 = vm->param(1);
         if ( elem1->isOrdinal() && elem2->isOrdinal() )
         {
            int64 num1 = elem1->forceInteger();
            int64 num2 = elem2->forceInteger();
            if ( num1 == num2 )
               vm->retval( num1 );
            else if ( num2 < num1 ) {
               int64 temp = num2;
               num2 = num1;
               num1 = temp;
            }
            num2 ++;

            vm->retval( (int64) (num1 + rng.randInt64(num2 - num1 - 1)) );
         }
         else
            vm->retval( *vm->param( rng.randInt(1) ) );
      break;

      default:
         vm->retval( *vm->param( rng.randInt(pcount - 1) ) );
   }
}

/*#
   @function randomChoice
   @brief Selects one of the arguments at random and returns it.
   @param ... At least two items of any kind.
   @return One of the parameters, picked at random.

   This function works as @a random when it receives more than two
   parameters, but its usage is not ambiguous in case there are two
   items from which to choice. The function raises an error
   if less than two parameters are passed.
*/

FALCON_FUNC  flc_randomChoice( ::Falcon::VMachine *vm )
{
   int32 pcount = vm->paramCount();

   switch( pcount )
   {
      case 0:
      case 1:
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime ).extra( "X,X,..." ) );
      break;

      default:
      {
         CoreObject *selfobj = vm->self().isNil() ? NULL : vm->self().asObject();
         MTRand &rng = selfobj ? ((RandomCarrier*)selfobj->getUserData())->getRNG() : vm->getRNG(); 

         vm->retval( *vm->param( rng.randInt(pcount - 1) ) );
      }
   }
}

/*#
   @function randomPick
   @brief Grabs repeatedly random elements from an array.
   @param series An array containing one or more items.
   @return One of the items in the array.
   @raise ParamError if the @b series is empty.

   This function choices one of the items contained in the @b series array
   at random.

   If the array is empty, a ParamError error is raised.
*/
FALCON_FUNC  flc_randomPick ( ::Falcon::VMachine *vm )
{
   Item *series = vm->param(0);
 

   if ( series == 0 || ! series->isArray() || series->asArray()->length() == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "A" ) );
   }

   CoreObject *selfobj = vm->self().isNil() ? NULL : vm->self().asObject();
   MTRand &rng = selfobj ? ((RandomCarrier*)selfobj->getUserData())->getRNG() : vm->getRNG();

   CoreArray &source = *series->asArray();
   vm->retval( source[ rng.randInt(source.length() - 1) ] );
}

/*#
   @function randomWalk
   @brief Performs a random walk in an array.
   @param series An array containing one or more items.
   @optparam size Desire size of the walk.
   @return An array built from randomly picked items.

   This function picks one or more elements from the given array,
   and stores them in a new array without removing them from the
   old one. Elements can be picked up repeatedly, and the size
   of the target array may be larger than the size of the original one.

   If the requested target size is zero, or if the original array is empty,
   an empty array is returned.

   If @b size is not given, 1 is assumed; if it's less than zero,
   then an the function will create an array of the same size of the
   @b series array, but the target array can contain multiple copies
   of the items in @b series, or it may be missing some of them.
*/
FALCON_FUNC  flc_randomWalk ( ::Falcon::VMachine *vm )
{
   Item *series = vm->param(0);
   Item *qty = vm->param(1);

   if ( series == 0 || ! series->isArray() 
      || (qty != 0 && ! qty->isOrdinal()) )  
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra( "A,[N]" ) );
   }

   CoreObject *selfobj = vm->self().isNil() ? NULL : vm->self().asObject();
   MTRand &rng = selfobj ? ((RandomCarrier*)selfobj->getUserData())->getRNG() : vm->getRNG(); 

   int32 number = qty == 0 ? 1 : (int32)qty->forceInteger();
   if( number < 0 ) number = series->asArray()->length();

   CoreArray *array = new CoreArray( number );
   CoreArray &source = *series->asArray();
   int32 slen = (int32) source.length();

   if ( slen > 0 ) {
      while( number > 0 ) {
         array->append( source[ rng.randInt(slen - 1) ] );
         number--;
      }
   }

   vm->retval( array );
}


/*#
   @function randomGrab
   @brief Grabs repeatedly random elements from an array.
   @param series An array from which items will be extracted.
   @optparam size Count of extracted items.
   @return An array with some or all of the items grabbed from the original elements.

   This function extracts a desired amount of items from the elements array,
   putting them in a new array that will be returned. Items left in the elements
   array have a fair chance to be selected and removed at every step. If the size
   parameter is greater or equal than the size of the elements array, the array is
   eventually emptied and all the items are moved to the new array, actually
   performing a complete fair shuffling of the original.

   If @b size is not given, 1 is assumed; if it's zero or less than zero,
   then all the elements in the @b series array will be taken.

   This function is suitable to emulate card shuffling or other random
   extraction events.
*/

FALCON_FUNC  flc_randomGrab ( ::Falcon::VMachine *vm )
{
   Item *series = vm->param(0);
   Item *qty = vm->param(1);

   if ( series == 0 || ! series->isArray() 
      || (qty != 0 && ! qty->isOrdinal()) )  
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra( "A,[N]" ) );
   }

   CoreObject *selfobj = vm->self().isNil() ? NULL : vm->self().asObject();
   MTRand &rng = selfobj ? ((RandomCarrier*)selfobj->getUserData())->getRNG() : vm->getRNG(); 
   
   int32 number = qty == 0 ? 1 : (int32)qty->forceInteger();
   if( number < 1 ) number = series->asArray()->length();

   CoreArray *array = new CoreArray( number );
   CoreArray &source = *series->asArray();
   int32 slen = (int32) source.length();

   while( number > 0 && slen > 0 ) {
      uint32 pos = rng.randInt(slen - 1);
      array->append( source[ pos ] );
      source.remove( pos );
      slen--;
      number--;
   }

   vm->retval( array );
}

/*#
   @function randomDice
   @brief Performs a virtual dice set trow.
   @param dices Number of dices to be thrown.
   @optparam sides Number of faces in the virtual dices.
   @return A random value which is the sum of the virtual throws.

   This function generates a series of successive @b dices throws,
   each one being integer value in the range [1, @b sides].

   If @b sides is not given, 6 is assumed.

   It would be easy to obtain the same result with simple instructions
   in Falcon, but this function spares several wasted VM cycles.

   The @b dices parameter must be greater than zero, and the
   and @b sides parameter must be greater than one.
*/

FALCON_FUNC  flc_randomDice( ::Falcon::VMachine *vm )
{
   Item *i_dices = vm->param(0);
   Item *i_sides = vm->param(1);

   if ( i_dices == 0 || ! i_dices->isOrdinal() || ( i_sides != 0 && ! i_sides->isOrdinal()) )  {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra( "N,N") );
   }

   CoreObject *selfobj = vm->self().isNil() ? NULL : vm->self().asObject();
   MTRand &rng = selfobj ? ((RandomCarrier*)selfobj->getUserData())->getRNG() : vm->getRNG(); 

   int64 dices = i_dices->forceInteger();
   int64 sides = i_sides == 0 ? 6 : i_sides->forceInteger();
   if( dices < 1 || sides < 2 )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ ).
         origin( e_orig_runtime ).
         extra( ">0,>1" ) );
   }

   int64 result = 0;
   for( int64 i = 0; i < dices; i ++ )
   {
      result += 1 + rng.randInt64(sides - 1);
   }

   vm->retval( result );
}

/*#
   @function randomSeed
   @brief Seeds the random number generator.
   @optparam seed An integer number being used as random seed.

   The random seed should be set once per program, possibly using a number that
   is likely to vary greatly among different executions. A good seed may be the
   return of the seconds() function, eventually multiplied by 1000 to make the
   milliseconds to take part in the seeding. If called without parameters, a number
   based on the current system timer value will be used.

   Repeated calls to random(), and calls based on random function as
   randomChoice, randomPick and so on, will produce the same sequences if
   randomSeed() is called with the same seed.

   Using a constant number as random seed may be a good strategy to produce
   predictable debug sequences.
*/
FALCON_FUNC  flc_randomSeed ( ::Falcon::VMachine *vm )
{
   Item *num = vm->param( 0 );
   uint32 value;

   if ( num == 0 )
   {
      value = (uint32) (Sys::_seconds() * 1000);
   }
   else
   {
      if ( ! num->isOrdinal() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra("N") );
      }

      value = (uint32) num->forceInteger();
   }

   CoreObject *selfobj = vm->self().isNil() ? NULL : vm->self().asObject();
   MTRand &rng = selfobj ? ((RandomCarrier*)selfobj->getUserData())->getRNG() : vm->getRNG(); 

   rng.seed( value );
}

}
}


/* end of random.cpp */
