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
#include <falcon/format.h>

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

FALCON_FUNC  mth_len ( ::Falcon::VMachine *vm )
{
   Item *elem;
   if ( ! vm->self().isMethodic() )
      elem = vm->param( 0 );
   else
      elem = &vm->self();

  if ( elem == 0 ) {
      vm->retval( 0 );
      return;
   }

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
         }
         
         vm->retval( (int64)num );
      }
      break;

      case FLC_ITEM_STRING:
      {
         String *cs = to_int->asString();
         int64 val;
         if ( ! cs->parseInt( val ) )
         {
            numeric nval;
            if ( cs->parseDouble( nval ) )
            {
               if ( nval > 9.223372036854775808e18 || nval < -9.223372036854775808e18 )
               {
                  vm->raiseRTError( new MathError( ErrorParam( e_domain, __LINE__ ) ) );
               }
               vm->retval( (int64) nval );
               return;
            }
            
            vm->raiseRTError( new ParseError( ErrorParam( e_numparse, __LINE__ ) ) );
         }
         vm->retval( val );
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
         numeric value;
         if ( ! cs->parseDouble( value ) )
         {
            
            vm->raiseRTError( new ParseError( ErrorParam( e_numparse, __LINE__ ) ) );
            return;
         }
         vm->retval( value );
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
FALCON_FUNC  mth_typeId ( ::Falcon::VMachine *vm )
{
   if ( vm->self().isMethodic() )
      vm->regA() = (int64) vm->self().dereference()->type();
   else {
      if ( vm->paramCount() > 0 )
         vm->regA() = (int64) vm->param(0)->type();
      else
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
   }
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
/*#
   @method isCallable BOM
   @brief Determines if an item is callable.
   @return true if the item is callable, false otheriwse.

   If the function returns true, then the call operator can be applied.
   If it returns false, the item is not a callable one, and trying to call
   it would cause an error.
*/

FALCON_FUNC  mth_isCallable ( ::Falcon::VMachine *vm )
{
   if ( vm->self().isMethodic() )
      vm->regA().setBoolean( vm->self().isCallable() );
   else {
      if ( vm->paramCount() > 0 )
         vm->regA().setBoolean( vm->param( 0 )->isCallable() ? 1 : 0 );
      else
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
   }
}

/*#
   @function getProperty
   @brief Returns the value of a property in an object.
   @param obj the source object, array or (blessed) dictionary.
   @param propName A string representing the name of a property or a method inside the object.
   @return the property
   @raise AccessError if the property can't be found.

   An item representing the property is returned. The returned value is
   actually a copy of the property; assigning a new value to it won't have any
   effect on the original object.

   If the property is a method, a callable method item is returned.
   If the property is not found, an error of class RangeError is raised.
*/

/*#
   @method getProperty Object
   @brief Returns the value of a property in an object.
   @param propName A string representing the name of a property or a method inside the object.
   @return the property
   @raise AccessError if the property can't be found.

   An item representing the property is returned. The returned value is
   actually a copy of the property; assigning a new value to it won't have any
   effect on the original object.

   If the property is a method, a callable method item is returned.
   If the property is not found, an error of class RangeError is raised.
*/
FALCON_FUNC  mth_getProperty( ::Falcon::VMachine *vm )
{
   Item *obj_x, *prop_x;
   if( vm->self().isMethodic() ) {
      obj_x = &vm->self();
      prop_x = vm->param(0);
   }
   else {
      obj_x= vm->param(0);
      prop_x = vm->param(1);
   }

   if ( obj_x == 0 || ! obj_x->isDeep() || prop_x == 0 || ! prop_x->isString() ) {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "O,S" ) ) );
   }

   obj_x->asDeepItem()->readProperty( *prop_x->asString(), vm->regA() );

   if ( vm->regA().isCallable() )
   {
      vm->regA().methodize( obj_x->asObjectSafe() );
   }
}

/*#
   @function setProperty
   @brief Sets the value of a proprety in a given object
   @param obj The source object.
   @param propName A string representing the name of a property or a method inside the object.
   @param value The property new value.
   @raise AccessError If the property can't be found.

   Alters the value of the property in the given object. If the required property is not present,
   an AccessError is raised.

*/
/*#
   @method setProperty Object
   @brief Sets the value of a proprety in this object
   @param propName A string representing the name of a property or a method inside the object.
   @param value The property new value.
   @raise AccessError If the property can't be found.

   Alters the value of the property in the given object. If the required property is not present,
   an AccessError is raised.

*/
/*#
   @method setProperty Array
   @brief Sets a binding (as a property) in the array.
   @param propName A string representing the name of a property or a method inside the array.
   @param value The property new value.
   @raise AccessError If the property can't be found.

   Alters the value of the property in the given array. If the required property is not present,
   an AccessError is raised.

*/
/*#
   @method setProperty Dictionary
   @brief Sets a property in dictionary based instances.
   @param propName A string representing the name of a property or a method inside the dictionary.
   @param value The property new value.
   @raise AccessError If the property can't be found.

   Alters the value of the property in the given dictionary. If the required property is not present,
   an AccessError is raised.

*/
FALCON_FUNC  mth_setProperty( ::Falcon::VMachine *vm )
{
   Item *obj_x, *prop_x, *new_item;
   if( vm->self().isMethodic() ) {
      obj_x = &vm->self();
      prop_x = vm->param(0);
      new_item = vm->param(1);
   }
   else {
      obj_x= vm->param(0);
      prop_x = vm->param(1);
      new_item = vm->param(2);
   }

   if ( obj_x == 0 || ! obj_x->isDeep() || prop_x == 0 || ! prop_x->isString() || new_item == 0) {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params )
         .extra( "O,S" ) ) );
   }

   obj_x->asDeepItem()->writeProperty( *prop_x->asString(), *new_item );
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

   CoreString *ret = new CoreString;
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
   @optparam format Specific object format.
   @return the string representation of the item.

   This function is useful to convert an unknown value in a string. The item may be any kind of Falcon
   item; the following rules apply:
      - Nil items are represented as "<NIL>"
      - Integers are converted in base 10.
      - Floating point values are converted in base 10 with a default precision of 6;
        numprec may be specified to change the default precision.
      - Array and dictionaries are represented as "Array of 'n' elements" or "Dictionary of 'n' elements".
      - Strings are copied.
      - Objects are represented as "Object 'name of class'", but if a toString() method is provided by the object,
        that one is called instead.
      - Classes and other kind of opaque items are rendered with their names.

   This function is not meant to provide complex applications with pretty-print facilities, but just to provide
   simple scripts with a simple and consistent output facility.

   If a @b format parameter is given, the format will be passed unparsed to toString() methods of underlying
   items.

   @see Format
*/

/*#
   @method toString BOM
   @brief Coverts the object to string.
   @optparam format Optional object-specific format string.

   Calling this BOM method is equivalent to call toString() core function
   passing the item as the first parameter.

   Returns a string representation of the given item. If applied on strings,
   it returns the string as is, while it converts numbers with default
   internal conversion. Ranges are represented as "[N:M:S]" where N and M are respectively
   lower and higher limits of the range, and S is the step. Nil values are represented as
   "Nil".

   The format parameter is not a Falcon format specification, but a specific optional
   object-specific format that may be passed to objects willing to use them.
   In example, the TimeStamp class uses this parameter to format its string
   representation.
*/

FALCON_FUNC  mth_ToString ( ::Falcon::VMachine *vm )
{
   Item *elem;
   Item *format;

   // methodic?
   if ( vm->self().isMethodic() )
   {
      elem = &vm->self();
      format = vm->param(0);
   }
   else {
      elem = vm->param(0);
      format = vm->param(1);
   }

   CoreString *target = new CoreString;

   if ( format != 0 )
   {
      if ( format->isString() )
      {
         Format fmt( *format->asString() );
         if ( ! fmt.isValid() )
         {
            throw new ParamError( ErrorParam( e_param_fmt_code ).
               extra( *format->asString() ) );
         }
         else
         {
            fmt.format( vm, *elem, *target );
         }
      }
      else
      {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( vm->self().isMethodic() ? "[S]" :  "X,[S]" ) ) );
         return;
      }
   }
   else {
      if ( vm->self().isMethodic() )
      {
         elem->toString( *target );
      }
      else
      {
         vm->itemToString( *target, elem );
      }
   }

   vm->retval( target );
}

/*#
   @method compare BOM
   @brief Performs a lexicographical comparison.
   @param item The item to which this object must be compared.
   @return -1, 0 or 1 depending on the comparation result.

   Performs a lexicographical comparison between the self item and the
   item passed as a parameter. If the item is found smaller than the parameter,
   it returns -1; if the item is greater than the parameter, it returns 1.
   If the two items are equal, it returns 0.

   The compare method, if overloaded, is used by the Virtual Machine to perform
   tests on unknown types (i.e. objects), and to sort dictionary keys.

   Item different by type are ordered by their type ID, as indicated in the
   documentation of the @a typeOf core function.

   By default, string comparison is performed in UNICODE character order,
   and objects, classes, vectors, and dictionaries are ordered by their
   internal pointer address.
*/
/*#
   @function compare
   @brief Performs a lexicographical comparison.
   @param operand1 The item to which this object must be compared.
   @param operand2 The item to which this object must be compared.
   @return -1, 0 or 1 depending on the comparation result.

   Performs a lexicographical comparison between the self item and the
   item passed as a parameter. If the item is found smaller than the parameter,
   it returns -1; if the item is greater than the parameter, it returns 1.
   If the two items are equal, it returns 0.

   The compare method, if overloaded, is used by the Virtual Machine to perform
   tests on unknown types (i.e. objects), and to sort dictionary keys.

   Item different by type are ordered by their type ID, as indicated in the
   documentation of the @a typeOf core function.

   By default, string comparison is performed in UNICODE character order,
   and objects, classes, vectors, and dictionaries are ordered by their
   internal pointer address.
*/

FALCON_FUNC mth_compare( VMachine *vm )
{
   Item *first;
   Item *second;

   if( vm->self().isMethodic() )
   {
      first = &vm->self();
      second = vm->param(0);
   }
   else
   {
      first = vm->param(0);
      second = vm->param(1);
   }

   if( first == 0 || second == 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( vm->self().isMethodic() ? "X" : "X,X" ) ) );
      return;
   }


   vm->retval( (int64) first->compare(*second) );
}


/*#
   @method clone BOM
   @brief Performs a deep copy of the item.
   @return A copy of the item.
   @raise CloneError if the item is not cloneable.

   Returns an item equal to the current one, but phisically separated.
   If the item is a sequence, only the first level of the item gets actually
   cloned: vectors and dictionaries gets cloned, but the items they store
   are just copied. This means that the new copy of the collection itself may
   change, and the older version will stay untouched, but if a deep item
   in the collection (as an object) is changed, its change will be reflected
   also in the original collection.

   Cloning objects causes each of their properties to be cloned. If they store
   an internal user data which is provided by extension modules or embedding
   applications, that data is cloned too. Behavior of user data is beyond the
   control of the script, and the data may actually be just referenced or
   it may also refuse to be cloned. In that case, this method will raise a
   CloneError, which indicates that a deep user data provided by an external
   module or application doesn't provide a cloning feature.

   @note Cloning objects that stores other objects referencing themselves in
   their properties may cause an endless loop in this version. To provide
   a safe duplicate of objects that may be organized in circular hierarcies,
   overload the clone method so that it creates a new instance of the item
   and just performs a flat copy of the properties.
*/

/*#
   @function clone
   @brief Performs a deep copy of the item.
   @param item The item to be copied.
   @return A copy of the item.
   @raise CloneError if the item is not cloneable.

   Returns an item equal to the @b item, but phisically separated.
   If the item is a sequence, only the first level of the item gets actually
   cloned: vectors and dictionaries gets cloned, but the items they store
   are just copied. This means that the new copy of the collection itself may
   change, and the older version will stay untouched, but if a deep item
   in the collection (as an object) is changed, its change will be reflected
   also in the original collection.

   Cloning objects causes each of their properties to be cloned. If they store
   an internal user data which is provided by extension modules or embedding
   applications, that data is cloned too. Behavior of user data is beyond the
   control of the script, and the data may actually be just referenced or
   it may also refuse to be cloned. In that case, this method will raise a
   CloneError, which indicates that a deep user data provided by an external
   module or application doesn't provide a cloning feature.

   @note Cloning objects that stores other objects referencing themselves in
   their properties may cause an endless loop in this version. To provide
   a safe duplicate of objects that may be organized in circular hierarcies,
   overload the clone method so that it creates a new instance of the item
   and just performs a flat copy of the properties.
*/

FALCON_FUNC mth_clone( VMachine *vm )
{
   bool result;
   if( vm->self().isMethodic() )
   {
      result = vm->self().clone( vm->regA() );
   }
   else
   {
      if( vm->paramCount() == 0 )
      {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra("X") ) );
         return;
      }
      else
      {
         result = vm->param(0)->clone( vm->regA() );
      }
   }

   if( ! result )
      vm->raiseError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) ) );
}

/*#
   @method className BOM
   @brief Returns the name of the class an instance is instantiated from.
   @return The class name of an object (a string) or nil.

   If applied to objects, returns the name of the class of which the object
   is an instance. When applied to classes, it return the class symbolic name.
   In all other cases, return nil.

   @see className
*/

/*#
   @function className
   @brief Returns the name of the class an instance is instantiated from.
   @param The item to be checked.
   @return The class name of an object (a string) or nil.

   If applied to objects, returns the name of the class of which the object
   is an instance. When applied to classes, it return the class symbolic name.
   In all other cases, return nil.

   @see BOM.className
*/

FALCON_FUNC mth_className( VMachine *vm )
{
   Item *self;

   if ( vm->self().isMethodic() )
   {
      self = &vm->self();
   }
   else {
      self = vm->param(0);
      if ( self == 0 )
      {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra("X") ) );
         return;
      }
   }

   switch( self->type() )
   {
      case FLC_ITEM_OBJECT:
         vm->retval(
            new CoreString(  vm->self().asObject()->generator()->symbol()->name() ) );
         break;

      case FLC_ITEM_CLASS:
         vm->retval(
            new CoreString(  vm->self().asClass()->symbol()->name() ) );
         break;

      default:
         vm->retnil();
   }

}

/*#
   @method baseClass BOM
   @brief Returns the class item from which an object has been instantiated.
   @return A class item or nil.

   If applied on objects, returns the class item that has been used
   to instantiate an object. Calling the returned item is equivalent
   to call the class that instantiated this object.

   The returned item can be used to create another instance of the same class,
   or for comparisons on @b select branches.

   If the item on which this method is applied is not an object, it returns nil.

   @see baseClass
*/

/*#
   @function baseClass
   @brief Returns the class item from which an object has been instantiated.
   @param item
   @return A class item or nil.

   If applied on objects, returns the class item that has been used
   to instantiate an object. Calling the returned item is equivalent
   to call the class that instantiated this object.

   The returned item can be used to create another instance of the same class,
   or for comparisons on @b select branches.

   If the item on which this method is applied is not an object, it returns nil.

   @see BOM.baseClass
*/

FALCON_FUNC mth_baseClass( VMachine *vm )
{
   Item *self;

   if ( vm->self().isMethodic() )
   {
      self = &vm->self();
   }
   else {
      self = vm->param(0);
      if ( self == 0 )
      {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra("X") ) );
         return;
      }
   }

   if( self->isObject() )
   {
      CoreClass* cls = const_cast<CoreClass*>(self->asObject()->generator());
      if ( cls != 0 )
      {
         vm->retval( cls );
         return;
      }
   }

   vm->retnil();
}

/*#
   @method derivedFrom BOM
   @brief Checks if this item has a given parent.
   @param cls A symbolic class name or a class instance.
   @return true if the given class is one of the ancestors of this item.

   If applied on objects, returns true if the given parameter is the name
   of the one of the classes that compose the class hierarchy of the object.

   If applied on class instances, it returns true if the parameter is its name
   or the name of one of its ancestors.

   In all the other cases, it return false.

   It is also possible to use directly the class instance as a parameter, instead of
   a class name. In example:

   @code
   object MyError from Error
       //...
   end

   > "Is MyError derived from 'Error' (by name)?: ", \
         MyError.derivedFrom( "Error" )

   > "Is MyError derived from 'Error' (by class)?: ", \
         MyError.derivedFrom( Error )
   @endcode

   @see derivedFrom
*/

/*#
   @function derivedFrom
   @brief Checks if this item has a given parent.
   @param item The item to be checked.
   @param cls A symbolic class name or a class instance.
   @return true if the given class is one of the ancestors of this item.

   If applied on objects, returns true if the given parameter is the name
   of the one of the classes that compose the class hierarchy of the object.

   If applied on class instances, it returns true if the parameter is its name
   or the name of one of its ancestors.

   In all the other cases, it return false.

   It is also possible to use directly the class instance as a parameter, instead of
   a class name. In example:

   @code
   object MyError from Error
       //...
   end

   > "Is MyError derived from 'Error' (by name)?: ", \
         derivedFrom( MyError, "Error" )

   > "Is MyError derived from 'Error' (by class)?: ", \
         derivedFrom( MyError, Error )
   @endcode

   @see BOM.derivedFrom
*/

FALCON_FUNC mth_derivedFrom( VMachine *vm )
{
   Item *i_clsName;
   Item *self;

   if( vm->self().isMethodic() )
   {
      self = &vm->self();
      i_clsName = vm->param( 0 );
   }
   else {
      self = vm->param(0);
      i_clsName = vm->param(1);
   }

   if( i_clsName == 0 || ! (i_clsName->isString() || i_clsName->isClass()) )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "S|C" ) ) );
      return;
   }

   const String *name;
   if ( i_clsName->isString() )
      name = i_clsName->asString();
   else
      name = &i_clsName->asClass()->symbol()->name();

   switch( self->type() )
   {
      case FLC_ITEM_OBJECT:
         vm->regA().setBoolean( (bool)self->asObjectSafe()->derivedFrom( *name ) );
         break;

      case FLC_ITEM_CLASS:
         vm->regA().setBoolean( (bool)self->asClass()->derivedFrom( *name ) );
         break;

      default:
         vm->regA().setBoolean( false );
   }
}

/*#
   @method source Method
   @brief Returns the object associated with this method.
   @return The object from which this method was created.

   Returns an object (or an item, in case this is a BOM method) that
   gave birth to this method.
*/
FALCON_FUNC Method_source( VMachine *vm )
{
   Item *self = vm->self().dereference();
   vm->retval( self->isMethod() ? self->asMethodItem() : *self );
}

/*#
   @method metaclass BOM
   @brief Returns the metaclass associated with this item.
   @return The metaclass of this item.
*/
/*#
   @function metaclass
   @brief Returns the metaclass associated with the given item.
   @param item The item of which the metaclass must be found.
   @return The metaclass of this item.
*/
FALCON_FUNC mth_metaclass( VMachine *vm )
{
   Item *self;

   if ( vm->self().isMethodic() )
   {
      self = &vm->self();
   }
   else {
      self = vm->param(0);
      if ( self == 0 )
      {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra("X") ) );
         return;
      }
   }

   CoreClass* cls = vm->getMetaClass( self->type() );
   fassert( cls );
   vm->retval( cls );
}



/*#
   @method ptr BOM
   @brief Returns a raw memory pointer out of this data (as an integer).
   @return An integer containing a pointer to this data.
   
   The default behavior of this method is to return the value of
   the memory location where inner data is stored, if the data is
   deep, and 0 otherwise. The Integer metaclass overrides this
   method so that it returns a dereferenced pointer, that is,
   the pointer value at the location indicated by the integer
   value, and String items overload this so that a pointer to
   the raw character sequence is returned.
*/
FALCON_FUNC BOM_ptr( VMachine *vm )
{
   if ( vm->self().isDeep() )
   {
      vm->retval( (int64) vm->self().asDeepItem() );
   }
   vm->retval( (int64) 0 );
}

/*#
   @method ptr Integer
   @brief Dereferences this integer as a pointer.
   @return Returns the pointer value at the location indicated in this integer.
   
   This function returns a pointer value (stored in a Falcon integer)
   taken from the memory location indicated by this integer.
*/
FALCON_FUNC Integer_ptr( VMachine *vm )
{
   void** ptr = (void**) vm->self().asInteger();
   vm->retval( (int64) *ptr );
}

/*#
   @method ptr GarbagePointer
   @brief Returns the inner data stored in this ponter.
   @return Deep data (as a pointer).
   
   This function returns a pointer value (stored in a Falcon integer)
   pointing to the inner FalconData served this garbage pointer.
*/
FALCON_FUNC GarbagePointer_ptr( VMachine *vm )
{
   vm->retval( (int64) vm->self().asGCPointer() );
}

/*#
   @method ptr String
   @brief Returns a pointer to raw data contained in this string.
   @return A string pointer.
   
   This function returns a pointer value (stored in a Falcon integer)
   pointing to the raw data in the string. The string is not encoded
   in any format, and its character size can be 1, 2 or 4 bytes per
   character depending on the values previusly stored. The string
   is granted to be terminated by an appropriate "\\0" value of the
   correct size. The value exists and is valid only while the original
   string (this item) stays unchanged.
*/

FALCON_FUNC String_ptr( VMachine *vm )
{
   String *str = vm->self().asString();
   str->c_ize();
   vm->retval( (int64) str->getRawStorage() );
}

/*#
   @method ptr MemBuf
   @brief Returns the pointer to the raw memory stored in this memory buffer.
   @return A memory pointer.
   
   This function returns a pointer (as a Falcon integer) to the memory
   area managed by this memory buffer.
*/

FALCON_FUNC MemoryBuffer_ptr( VMachine *vm )
{
   vm->retval( (int64) vm->self().asMemBuf()->data() );
}

}
}
