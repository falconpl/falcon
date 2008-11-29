/*
   FALCON - The Falcon Programming Language.
   FILE: item_ext.cpp

   Generic item handling
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/attribute.h>
#include <falcon/fbom.h>

/*#

*/

namespace Falcon {
namespace core {

/*#
   @funset generic_item_api Type mangling
   @brief Functions managing item conversions, type detection, item structure
      management and generically manipulating Falcon items.

   @beginset generic_item_api
*/

/*#
   @function len
   @brief Retreives the lenght of a collection
   @param item an item of any kind
   @return the count of items in the sequence, or 0.

   The returned value represent the "size" of the item passed as a parameter.
   The number is consistent with the object type: in case of a string, it
   represents the count of characters, in case of arrays or dictionaries it
   represents the number of elements, in all the other cases the returned
   value is 0.
*/

FALCON_FUNC  len ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 ) {
      vm->retval( 0 );
      return;
   }

   Item *elem = vm->param(0);
   switch( elem->type() ) {
      case FLC_ITEM_STRING:
         vm->retval( (int64) elem->asString()->length() );
      break;

      case FLC_ITEM_ARRAY:
         vm->retval( (int64) elem->asArray()->length() );
      break;

      case FLC_ITEM_MEMBUF:
         vm->retval( (int64) elem->asMemBuf()->length() );
      break;

      case FLC_ITEM_DICT:
         vm->retval( (int64) elem->asDict()->length() );
      break;

      case FLC_ITEM_ATTRIBUTE:
         vm->retval( (int64) elem->asAttribute()->size() );
      break;

      case FLC_ITEM_RANGE:
         vm->retval( 3 );
      break;

      default:
         vm->retval( 0 );
   }
}


/*#
   @function int
   @brief Converts the given parameter to integer.
   @param item The item to be converted
   @return An integer value.
   @raise ParseError in case the given string cannot be converted to an integer.
   @raise MathError if a given floating point value is too large to be converted to an integer.

   Integer values are just copied. Floating point values are converted to long integer;
   in case they are too big to be prepresented a RangeError is raised.
   Strings are converted from base 10. If the string cannot be converted,
   or if the value is anything else, a MathError instance is raised.
*/
FALCON_FUNC  val_int ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 ) {
      vm->retnil();
      return;
   }

   Item *to_int = vm->param(0);

   switch( to_int->type() ) {
      case FLC_ITEM_INT:
          vm->retval( to_int->asInteger() );
      break;

      case FLC_ITEM_NUM:
      {
         numeric num = to_int->asNumeric();
         if ( num > 9.223372036854775808e18 || num < -9.223372036854775808e18 )
         {
            vm->raiseRTError( new MathError( ErrorParam( e_domain, __LINE__ ) ) );
            return;
         }
         vm->retval( (int64)num );
      }
      break;

      case FLC_ITEM_STRING:
      {
         String *cs = to_int->asString();
         if ( cs->size() == 0 )
            vm->retval(0);
         else {
            int32 pos = cs->size() -1;
            if ( pos > 18 ) {
               vm->raiseRTError( new ParseError( ErrorParam( e_numparse_long, __LINE__ ) ) );
               return;
            }
            uint32 chr =  cs->getCharAt( pos );
            uint64 val = 0;
            uint64 base = 1;
            while( pos > 0 ) {
               if ( chr < '0' || chr > '9' ) {
                  vm->raiseRTError( new ParseError( ErrorParam( e_numparse, __LINE__ ) ) );
                  return;
               }
               val += ( chr -'0') * base;
               pos--;
               chr =  cs->getCharAt( pos );
               base *= 10;
            }
            if ( chr == '-' )
               vm->retval( -(int64)val );
            else {
               if ( chr < '0' || chr > '9' ) {
                  vm->raiseRTError( new ParseError( ErrorParam( e_numparse, __LINE__ ) ) );
                  return;
               }

               vm->retval( (int64)(val + ( chr -'0' ) * base ) );
            }
         }
      }
      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "N|S" ) ) );
   }
}

/*#
   @function numeric
   @brief Converts the given parameter to numeric.
   @param item The item to be converted
   @return A numeric value.
   @raise ParseError in case the given string cannot be converted to an integer.
   @raise MathError if a given floating point value is too large to be converted to an integer.

   Floating point values are just copied. Integer values are converted to floating point;
   in case of very large integers, precision may be lost.
   Strings are converted from base 10. If the string cannot be converted,
   or if the value is anything else, a MathError instance is raised.
*/
FALCON_FUNC  val_numeric ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 ) {
      vm->retnil();
      return;
   }

   Item *to_numeric = vm->param(0);

   switch( to_numeric->type() ) {
      case FLC_ITEM_NUM:
          vm->retval( to_numeric->asNumeric() );
      break;

      case FLC_ITEM_INT:
      {
         int64 num = to_numeric->asInteger();
         vm->retval( (numeric)num );
      }
      break;

      case FLC_ITEM_STRING:
      {
         String *cs = to_numeric->asString();
         if ( cs->size() == 0 )
            vm->retval(0);
         else {
            int32 pos = cs->size() -1;
            if ( pos > 18 ) {
               vm->raiseRTError( new MathError( ErrorParam( e_numparse_long, __LINE__ ) ) );
               return;
            }
            uint32 chr =  cs->getCharAt( pos );
            numeric val = 0;
            uint32 base = 1;
            while( pos > 0 ) {
               if ( chr == '.' ) {
                  numeric decbase = 1 / (numeric) base;
                  val *= decbase;

                  pos--;
                  chr = cs->getCharAt( pos );
                  base = 1;
                  continue;
               }
               else if ( chr < '0' || chr > '9' ) {
                  vm->raiseRTError( new MathError( ErrorParam( e_numparse, __LINE__ ) ) );
                  return;
               }
               val += ( chr -'0' ) * base;
               pos--;
               chr =  cs->getCharAt( pos );
               base *= 10;
            }

            if ( chr == '-' )
               vm->retval( -(numeric)val );
            else {
               if ( chr < '0' || chr > '9' ) {
                  vm->raiseRTError( new MathError( ErrorParam( e_numparse, __LINE__ ) ) );
                  return;
               }

               vm->retval( (numeric)(val + ( chr -'0' ) * base ) );
            }
         }
      }
      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "(N|S)" ) ) );
   }
}

/*#
   @function typeOf
   @param item An item of any kind.
   @brief Returns an integer indicating the type of an item.
   @return A constant indicating the type of the item.

   The typeId returned is an integer; the Falcon compiler is fed with a set of
   compile time constants that can be used to determine the type of an item.
   Those constants are always available at Falcon sources.

   The value returned may be one of the following:
   - @b NilType - the item is NIL
   - @b BooleanType - the item is true or false
   - @b IntegerType - the item is an integer
   - @b NumericType - the item is a floating point number
   - @b RangeType - the item is a range (a pair of two integers)
   - @b FunctionType - the item is a function
   - @b StringType - the item is a string
   - @b LBindType - the item is a late binding symbol
   - @b MemBufType - the item is a Memory Buffer Table
   - @b ArrayType - the item is an array
   - @b DictionaryType - the item is a dictionary
   - @b ObjectType - the item is an object
   - @b ClassType - the item is a class
   - @b MethodType - the item is a method
   - @b ClassMethodType - the item is a method inside a class
*/
FALCON_FUNC  typeOf ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 )
      vm->retnil();
   else
      vm->retval( vm->param( 0 )->type() );
}

/*#
   @function isCallable
   @brief Determines if an item is callable.
   @inset generic_item_api
   @param item The item to be converted
   @return true if the item is callable, false otheriwse.

   If the function returns true, then the call operator can be applied.
   If it returns false, the item is not a callable one, and trying to call
   it would cause an error.
*/

FALCON_FUNC  isCallable ( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() == 0 )
      vm->retval( 0 );
   else
      vm->retval( vm->param( 0 )->isCallable() ? 1 : 0 );
}

/*#
   @function getProperty
   @brief Returns the value of a property in an object.
   @param obj the source object
   @param propName A string representing the name of a property or a method inside the object.
   @return the property
   @raise AccessError if the property can't be found.

   An item representing the property is returned. The returned value is
   actually a copy of the property; assigning a new value to it won't have any
   effect on the original object.

   If the property is a method, a callable method item is returned.
   If the property is not found, an error of class RangeError is raised.
*/
FALCON_FUNC  getProperty( ::Falcon::VMachine *vm )
{
   Item *obj_x = vm->param(0);
   Item *prop_x = vm->param(1);

   if ( obj_x == 0 || ! obj_x->isObject() || prop_x == 0 || ! prop_x->isString() ) {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "(0,S)" ) ) );
   }
   else if ( ! obj_x->asObjectSafe()->getProperty( *prop_x->asString(), vm->regA() ) )
   {
      vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
   }

   if ( vm->regA().isCallable() )
   {
      vm->regA().methodize( obj_x->asObjectSafe() );
   }
}

/*#
   @function setProperty
   @param obj The source object.
   @param propName A string representing the name of a property or a method inside the object.
   @param value The property new value.
   @raise AccessError If the property can't be found.

   Alters the value of the property in the given object. If the required property is not present,
   an AccessError is raised.

*/
FALCON_FUNC  setProperty( ::Falcon::VMachine *vm )
{
   Item *obj_x = vm->param(0);
   Item *prop_x = vm->param(1);
   Item *new_item = vm->param(2);

   if ( obj_x == 0 || ! obj_x->isObject() || prop_x == 0 || ! prop_x->isString() || new_item == 0) {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params )
         .extra( "O,S" ) ) );
   }
   else if ( ! obj_x->asObjectSafe()->setProperty( *prop_x->asString(), *new_item ) )
   {
      vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
   }
}

/*#
   @function chr
   @brief Returns a string containing a single character that corresponds to the given number.
   @inset generic_item_api
   @param number Numeric code of the desired character
   @return a single-char string.

   This function returns a single character string whose only character is the UNICODE
   equivalent for the given number. The number must be a valid UNICODE character,
   so it must be in range 0-0xFFFFFFFF.
*/

FALCON_FUNC  chr ( ::Falcon::VMachine *vm )
{
   uint32 val;
   Item *elem = vm->param(0);
   if ( elem == 0 ) return;
   if ( elem->type() == FLC_ITEM_INT )
      val = (uint32) elem->asInteger();
   else if ( elem->type() == FLC_ITEM_NUM )
      val = (uint32) elem->asNumeric();
   else {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "N" ) ) );
      return;
   }

   String *ret = new GarbageString( vm );
   ret->append( val );
   vm->retval( ret );
}

/*#
   @function ord
   @brief Returns the numeric UNICODE ID of a given character.
   @inset generic_item_api
   @param string The character for which the ID is requested.
   @return the UNICODE value of the first element in the string.

   The first character in string is taken, and it's numeric ID is returned.

   @see chr
*/
FALCON_FUNC  ord ( ::Falcon::VMachine *vm )
{
   Item *elem = vm->param(0);
   if ( elem == 0 || ! elem->isString() || elem->asString()->size() == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params).extra( "S" ) ) );
      return;
   }

   vm->retval( (int64) elem->asString()->getCharAt(0) );
}

/*#
   @function toString
   @brief Returns a string representation of the item.
   @param item The item to be converted to string.
   @optparam numprec Number of significative decimals for numeric items.
   @return the string representation of the item.

   This function is useful to convert an unknown value in a string. The item may be any kind of Falcon
   item; the following rules apply:
      - Nil items are represented as “<NIL>”
      - Integers are converted in base 10.
      - Floating point values are converted in base 10 with a default precision of 6;
        numprec may be specified to change the default precision.
      - Array and dictionaries are represented as “Array of 'n' elements” or “Dictionary of 'n' elements”.
      - Strings are copied.
      - Objects are represented as “Object 'name of class'”, but if a toString() method is provided by the object,
        that one is called instead.
      - Classes and other kind of opaque items are rendered with their names.

   This function is not meant to provide complex applications with pretty-print facilities, but just to provide
   simple scripts with a simple and consistent output facility.

   @see Format
*/

FALCON_FUNC  hToString ( ::Falcon::VMachine *vm )
{
   Item *elem = vm->param(0);
   Item *format = vm->param(1);

   Fbom::toString( vm, elem, format );
}

}
}
