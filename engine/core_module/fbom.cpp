/*
   FALCON - The Falcon Programming Language
   FILE: fbom.cpp

   Falcon basic object model
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer lug 4 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon basic object model
*/

#include <falcon/fbom.h>
#include <falcon/types.h>
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/vm.h>
#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/attribute.h>
#include <falcon/membuf.h>
#include <falcon/coretable.h>

#include <falcon/format.h>

#include <falcon/autocstring.h>
#include <falcon/bommap.h>

#include <stdio.h>

/*#
   @beginmodule core
*/

/*#
   @class BOM Basic object model
   @brief Methods common to all Falcon items.

   Falcon provides a basic object model, (short BOM), which is namely a
   collection of predefined methods that can be applied on any Falcon
   item, be it an object, or any other basic type.

   BOM methods may be overloaded by classes and objects; the only difference
   between BOM and ordinary methods is that, while every item provides them,
   an explicit test with provides or in operators will fail.
   In example, an object must explicitly provide toString() overloading for the
   operator @b provides to report success; testing “item provides toString”
   on an object (or any other item) will fail unless the method is explicitly
   defined by the tested object.

   BOM is sparingly used also by the Virtual Machine. In example, key sorting
   in dictionaries and relational operators uses the “compare()” BOM method.
   A class may define a personalized ordering by redefining the compare method.
   Also, toString() is often used by the VM to provide consistent output.

   Currently, BOM is undifferentiated. Future extension may provide type specific
   methods.
*/


namespace Falcon {
/*#
   @method toString BOM
   @brief Coverts the object to string.
   @optparam format Optional object-specific format string.

   Calling this BOM method is equivalent to call toString() core function
   passing the item as the first parameter.

   Returns a string representation of the given item. If applied on strings,
   it returns the string as is, while it converts numbers with default
   internal conversion. Ranges are represented as “[N:M:S]” where N and M are respectively
   lower and higher limits of the range, and S is the step. Nil values are represented as
   “Nil”.

   The format parameter is not a Falcon format specification, but a specific optional
   object-specific format that may be passed to objects willing to use them.
   In example, the TimeStamp class uses this parameter to format its string
   representation.
*/

/* BOMID: 0 */
FALCON_FUNC BOM_toString( VMachine *vm )
{
   Fbom::toString( vm, &vm->self(), vm->bomParam( 0 ) );
}

/*#
   @method len BOM
   @brief Gets the length of a sequence.
   @return sequence lenght or 0 if the item is not a sequence.

   Equivalent to the Core @a len function. Applied on collections
   and strings, it returns the count of items in the collection,
   or the number of characters in the string.

   Applied to other values, it returns 0.
*/

/* BOMID: 1 */
FALCON_FUNC BOM_len( VMachine *vm )
{
   Item *elem = &vm->self();
   switch( elem->type() ) {
      case FLC_ITEM_STRING:
         vm->retval( (int64) elem->asString()->length() );
      break;

      case FLC_ITEM_MEMBUF:
         vm->retval( (int64) elem->asMemBuf()->length() );
      break;

      case FLC_ITEM_ARRAY:
         vm->retval( (int64) elem->asArray()->length() );
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
   @method first BOM
   @brief Returns an iterator to the first element of a collection
   @return An iterator.
   @raise AccessError if the item is not a collection.

   If the item is an array, a dictionary or a string, it returns an iterator
   pointing to the first element of the collection. If it's a range, it returns
   the begin of the range. In every other case, it raises an AccessError.
*/

/* BOMID: 2 */
FALCON_FUNC BOM_first( VMachine *vm )
{
   const Item &self = vm->self();
   switch( self.type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_ARRAY:
      case FLC_ITEM_DICT:
      case FLC_ITEM_ATTRIBUTE:
         Fbom::makeIterator( vm, self, true );
      break;

      case FLC_ITEM_RANGE:
         vm->retval( (int64) self.asRangeStart() );
      break;

      case FLC_ITEM_LBIND:
         vm->retval( new GarbageString( vm, *self.asLBind() ) );
      break;

      default:
         vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
   }
}

/*#
   @method last BOM
   @brief Returns an iterator to the last element of a collection
   @return An iterator.
   @raise AccessError if the item is not a collection.

   If the item is an array, a dictionary or a string, it returns an iterator
   pointing to the last element of the collection. If it's a range, it returns
   the end of the range, or nil if the range is open.
   In every other case, it raises an AccessError.
*/

/* BOMID: 3 */
FALCON_FUNC BOM_last( VMachine *vm )
{
   const Item &self = vm->self();

   switch( self.type() )
   {
      case FLC_ITEM_STRING:
      case FLC_ITEM_MEMBUF:
      case FLC_ITEM_ARRAY:
      case FLC_ITEM_DICT:
      // attributes cannot be scanned backwards
         Fbom::makeIterator( vm, self, false );
      break;

      case FLC_ITEM_RANGE:
         if( self.asRangeIsOpen() )
            vm->retnil();
         else
            vm->retval( (int64) self.asRangeEnd() );
      break;

      case FLC_ITEM_LBIND:
         vm->retval( self.isFutureBind() ? self.asFutureBind() : Item() );
      break;

      default:
         vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
   }
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

/* BOMID: 4 */
FALCON_FUNC BOM_compare( VMachine *vm )
{
   if( vm->bomParamCount() == 0 )
   {
       vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
       return;
   }

   Item *comparand = vm->bomParam( 0 );
   vm->retval( vm->self().compare( *comparand ) );

}

/*#
   @method equal BOM
   @brief Checks for item equality.
   @param item The item to which this object must be compared.
   @return true if the two items are considered equal, false otherwise.

   Returns true when two items are found identical. It is internally
   implemented using the @a BOM.compare method, so it is sufficient to
   overload the compare method so that it returns 0 when two items are
   found identical.
*/

/* BOMID: 5 */
FALCON_FUNC BOM_equal( VMachine *vm )
{
   if( vm->bomParamCount() == 0 )
   {
       vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X" ) ) );
       return;
   }

   Item *comparand = vm->bomParam( 0 );
   vm->regA().setBoolean( vm->self().equal( *comparand ) );

}

/*#
   @method type BOM
   @brief Returns an integer representing the type of the item.
   @return The type of the item.

   Returns the typeID of this object; it is equivalent to call the core @a typeOf function.
*/

/* BOMID: 6 */
FALCON_FUNC BOM_type( VMachine *vm )
{
   vm->retval( vm->self().type() );
}

/*#
   @method className BOM
   @brief Returns the name of the class an object is derived from.
   @return The class name of an object or nil.

   If applied to objects, returns the name of the class of which the object
   is an instance. When applied to classes, it return the class symbolic name.
   In all other cases, return nil.
*/

/* BOMID: 7 */
FALCON_FUNC BOM_className( VMachine *vm )
{
   if( vm->self().isObject() )
   {
      vm->retval(
         new GarbageString( vm, vm->self().asObject()->instanceOf()->name() ) );
   }
   else if( vm->self().isClass() )
   {
      vm->retval(
         new GarbageString( vm, vm->self().asClass()->symbol()->name() ) );
   }
   else {
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
*/

/* BOMID: 8 */
FALCON_FUNC BOM_baseClass( VMachine *vm )
{

   if( vm->self().isObject() )
   {
      Symbol *cls = vm->self().asObject()->instanceOf();
      Item *i_cls = vm->findLocalSymbolItem( cls->name() );

      if( i_cls != 0 )
         vm->retval( *i_cls );
      else
         vm->retnil();
   }
   else {
      vm->retnil();
   }
}

/*#
   @method derivedFrom BOM
   @brief Checks if this item has a given parent.
   @param class A symbolic class name or a class instance.
   @return true if the given class is one of the ancestors of this item.

   If applied on objects, returns true if the given parameter is the name
   of the one of the classes that compose the class hierarchy of the object.

   If applied on class items, it returns true if the parameter is its name
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

*/

/* BOMID: 9 */
FALCON_FUNC BOM_derivedFrom( VMachine *vm )
{
   Item *i_clsName = vm->bomParam( 0 );

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

   if( vm->self().isObject() )
   {
      vm->regA().setBoolean( (bool)vm->self().asObject()->derivedFrom( *name ) );
   }
   else if( vm->self().isClass() )
   {
      vm->regA().setBoolean( (bool)vm->self().asClass()->derivedFrom( *name ) );
   }
   else {
      vm->regA().setBoolean( false );
   }
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

/* BOMID: 10 */
FALCON_FUNC BOM_clone( VMachine *vm )
{
   if ( ! vm->self().clone( vm->regA(), vm ) )
   {
      vm->raiseError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }
}

/*#
   @method serialize BOM
   @brief Serialize the item on a stream for persistent storage.
   @param stream The stream on which to perform serialization.
   @raise IoError on stream errors.

   The item is stored on the stream so that a deserialize() call on the same
   position in the stream where serialization took place will create
   an exact copy of the serialized item.

   The application must ensure that the item does not contains circular references,
   or the serialization will enter an endless loop.

   In case the underlying stream write causes an i/o failure, an error is raised.
*/

/* BOMID: 11 */
FALCON_FUNC BOM_serialize( VMachine *vm )
{
   Item *fileId = vm->bomParam( 0 );
   Item *source = vm->self().dereference();

   if( fileId == 0 || ! fileId->isObject() || ! fileId->asObject()->derivedFrom( "Stream" ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).
         extra( "O:Stream" ) ) );
      return;
   }

   Stream *file = (Stream *) fileId->asObject()->getUserData();
   Item::e_sercode sc = source->serialize( file );
   switch( sc )
   {
      case Item::sc_ok: vm->retval( 1 ); break;
      case Item::sc_ferror:
         vm->raiseModError( new IoError( ErrorParam( e_modio, __LINE__ ).origin( e_orig_vm ) ) );
      default:
         vm->retnil(); // VM may already have raised an error.
         //TODO: repeat error.
   }

}

/*#
   @method attribs BOM
   @brief Returns an array of attributes being given to a certain object.
   @return An array of attributes or nil.
   @raise IoError on stream errors.

   This method can be used to inspect which attributes are given to a certain object.
   If the method is called on a non-object item, nil is returned. If the object
   hasn't any attribute, an empty array is returned.
*/

/* BOMID: 12 */
FALCON_FUNC BOM_attribs( VMachine *vm )
{
   Item *source = vm->self().dereference();
   if( source->isObject() )
   {
      CoreArray *array = new CoreArray( vm );
      AttribHandler *attribs = source->asObject()->attributes();
      while( attribs != 0 )
      {
         array->append( attribs->attrib() );
         attribs = attribs->next();
      }
      vm->retval( array );
   }
   else
      vm->retnil();
}

/*#
   @method backTrim BOM
   @brief Trims trailing whitespaces in a string.
   @return The trimmed version of the string.
   @raise AccessError if the given item is not a string.

   This method removes whitespaces from the end of a string
   and returns the trimmed version.
*/

/* BOMID: 13 */
FALCON_FUNC BOM_backTrim( VMachine *vm )
{
   const Item &self = vm->self();
   if ( self.isString() )
   {
      GarbageString *s = new GarbageString( vm, *self.asString() );
      s->backTrim();
      vm->retval( s );
   } else {
      vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
   }
}

/*#
   @method frontTrim BOM
   @brief Trims front whitespaces in a string.
   @return The trimmed version of the string.
   @raise AccessError if the given item is not a string.

   This method removes whitespaces from the beginning of a string
   and returns the trimmed version.
*/

/* BOMID: 14 */
FALCON_FUNC BOM_frontTrim( VMachine *vm )
{
   const Item &self = vm->self();
   if ( self.isString() )
   {
      GarbageString *s = new GarbageString( vm, *self.asString() );
      s->frontTrim();
      vm->retval( s );
   } else {
      vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
   }
}

/*#
   @method allTrim BOM
   @brief Trims whitespaces from both ends of a string.
   @return The trimmed version of the string.
   @raise AccessError if the given item is not a string.

   This method removes whitespaces from the beginning and the end
   of a string and returns the trimmed version.
*/

/* BOMID: 15 */
FALCON_FUNC BOM_allTrim( VMachine *vm )
{
   const Item &self = vm->self();
   if ( self.isString() )
   {
      GarbageString *s = new GarbageString( vm, *self.asString() );
      s->trim();
      vm->retval( s );
   } else {
      vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
   }
}

/*#
   @method front BOM
   @brief Returns the first element of a collection.
   @optparam bRemove If true, remove also the element.
   @optparam bNumeric If true, returns a character value instead of a string.
   @return The first element.
   @raise AccessError if the item is not a collection, or if the collection is empty.

   This BOM method retreives the first element of a collection. It translates
   into an array accessor with [0] in case of arrays. This method is particularly
   useful to deal with collections that cannot be directly accessed by item
   number, as i.e. dictionaries.

   If the sequence is a string, then a second parameter may be provided; if it's true,
   the UNICODE value of the string character will be returned, otherwise the caller
   will receive a string containing the desired character.

   If the sequence is a dictionary, normally, only the value will be returned.
   If bNumOrKey is true, then the coresponding key will be returned instead.

   Also, this method is relatively efficient with respect to both direct access and
   iterator instantation.
*/

/* BOMID: 16 */
FALCON_FUNC BOM_front( VMachine *vm )
{
   const Item &self = vm->self();
   Item *i_remove = vm->bomParam( 0 );
   bool bRemove = i_remove == 0 ? false : i_remove->isTrue();

   switch( self.type() )
   {
      case FLC_ITEM_ARRAY:
      {
         CoreArray *array = self.asArray();
         if ( array->length() != 0 )
         {
            vm->retval( array->at(0) );
            if( bRemove )
               array->remove(0);
           return;
         }
      }
      break;

      case FLC_ITEM_STRING:
      {
         String *str = self.asString();
         if ( str->size() != 0 )
         {
            Item *i_numeric = vm->bomParam( 1 );
            if ( i_numeric == 0 || i_numeric == false )
               vm->retval( str->subString(0,1) );
            else
               vm->retval( (int64) str->getCharAt( 0 ) );

            if( bRemove )
               str->remove( 0, 1 );
           return;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *dict = self.asDict();
         if ( ! dict->empty() )
         {
            DictIterator *iter = dict->first();
            Item *i_numeric = vm->bomParam( 1 );
            if ( i_numeric == 0 || ! i_numeric->isTrue() )
               vm->retval( iter->getCurrent() );
            else
               vm->retval( iter->getCurrentKey() );
            if ( bRemove )
               dict->remove( *iter );
            delete iter;
            return;
         }
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = self.asMemBuf();
         if ( mb->size() > 0 && ! bRemove )
         {
            vm->retval( (int64) mb->get( 0 ) );
            return;
         }
      }
      break;
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
}

/*#
   @method back BOM
   @brief Returns the last element of a collection.
   @optparam bRemove If true, remove also the element.
   @optparam bNumOrKey If true, returns a character value instead of a string,
      or a key instead of a value.
   @return The first element.
   @raise AccessError if the item is not a collection.

   This BOM method retreives the first element of a collection. It translates
   into an array accessor with [0] in case of arrays. This method is particularly
   useful to deal with collections that cannot be directly accessed by item
   number, as i.e. dictionaries.

   If the sequence is a string, then a second parameter may be provided; if it's true,
   the UNICODE value of the string character will be returned, otherwise the caller
   will receive a string containing the desired character.

   If the sequence is a dictionary, normally, only the value will be returned.
   If bNumOrKey is true, then the coresponding key will be returned instead.

   Also, this method is relatively efficient with respect to both direct access and
   iterator instantation.
*/

/* BOMID: 17 */
FALCON_FUNC BOM_back( VMachine *vm )
{
   const Item &self = vm->self();
   Item *i_remove = vm->bomParam( 0 );
   bool bRemove = i_remove == 0 ? false : i_remove->isTrue();

   switch( self.type() )
   {
      case FLC_ITEM_ARRAY:
      {
         CoreArray *array = self.asArray();
         if ( array->length() != 0 )
         {
            vm->retval( array->at(array->length()-1) );
            if( bRemove )
               array->remove(array->length()-1);
           return;
         }
      }
      break;

      case FLC_ITEM_STRING:
      {
         String *str = self.asString();
         if ( str->size() != 0 )
         {
            Item *i_numeric = vm->bomParam( 1 );
            if ( i_numeric == 0 || i_numeric == false )
               vm->retval( str->subString( str->length()-1, str->length()) );
            else
               vm->retval( (int64) str->getCharAt( str->length()-1 ) );

            if( bRemove )
               str->remove( str->length()-1, 1 );
           return;
         }
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *dict = self.asDict();
         if ( ! dict->empty() )
         {
            DictIterator *iter = dict->last();
            Item *i_numeric = vm->bomParam( 1 );
            if ( i_numeric == 0 || ! i_numeric->isTrue() )
               vm->retval( iter->getCurrent() );
            else
               vm->retval( iter->getCurrentKey() );

            if ( bRemove )
               dict->remove( *iter );
            delete iter;
            return;
         }
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = self.asMemBuf();
         if ( mb->size() > 0 && ! bRemove )
         {
            vm->retval( (int64) mb->get( mb->length()-1 ) );
            return;
         }
      }
      break;
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
}

/*#
   @method table BOM
   @brief Returns the table related with this array.
   @raise AccessError if the item is not an array.
   @return The table of which this item is a row.

   This BOM method retreives the table that is related with the
   item, provided the item is an array being part of a table.

   In case the item is an array, but it doesn't belong to any
   table, nil is returned.
*/

/* BOMID: 18 */
FALCON_FUNC BOM_table( VMachine *vm )
{
   const Item &self = vm->self();

   if ( self.isArray() )
   {
      CoreArray *array = self.asArray();
      if ( array->table() != 0 )
      {
         vm->retval( array->table() );
         return;
      }

      vm->retnil();
      return;
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
}

/*#
   @method tabField BOM
   @brief Returns one of the items in the array, given the field name.
   @param field The field name or position to be retreived.
   @raise AccessError if the item is not an array.
   @return An item in the array or the default column value.

   If this item is an array and is part of a table, the field with
   the given name or ID (number) is searched in the table definition,
   and if found, it is returned. If the coresponding item in the array
   is nil, then the table column data (default data) is returned instead,
   unless the item is also an OOB item. In that case, nil is returned
   and the default column value is ignored.
*/

/* BOMID: 19 */
FALCON_FUNC BOM_tabField( VMachine *vm )
{
   const Item &self = vm->self();
   Item *i_field = vm->bomParam( 0 );
   if ( i_field == 0 ||
      ! ( i_field->isString() || i_field->isOrdinal() ))
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).
               extra( "(N)" ) ) );
      return;
   }

   if ( self.isArray() )
   {
      CoreArray *array = self.asArray();
      if ( array->table() != 0 )
      {
         // if a field paramter is given, return its value
         uint32 num = (uint32) CoreTable::noitem;
         CoreTable *table = reinterpret_cast<CoreTable*>(array->table()->getUserData());

         if ( i_field->isString() )
         {
            num = table->getHeaderPos( *i_field->asString() );
         }
         else {
            // we already checked, must be a field
            num = (uint32) i_field->forceInteger();
         }

         if ( num < array->length() )
         {
            Item &data = (*array)[num];
            if ( ! data.isNil() )
            {
               vm->retval( data );
            }
            else {
               if ( data.isOob() )
                  vm->retnil();
               else
                  vm->retval( table->columnData( num ) );
            }

            return;
         }

         vm->raiseRTError( new AccessError( ErrorParam( e_param_range ) ) );
         return;
      }

      vm->retnil();
      return;
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
}

/*#
   @method tabRow BOM
   @brief Returns the row ID of this element.
   @raise AccessError if the item is not a table row (array).
   @return A number indicating the position of this row in the table, or
      nil if this item is an array, but it's not stored in a table.

   This method returns the position of this element in a table.

   This number gets valorized only after a @a Table.get or @a Table.find
   method call, so that it is possible to know what index had this element
   in the owning table. If the table is changed by inserting or removing
   a row, the number returned by this function becomes meaningless.
*/

/* BOMID: 20 */
FALCON_FUNC BOM_tabRow( VMachine *vm )
{
   const Item &self = vm->self();

   if ( self.isArray() )
   {
      CoreArray *array = self.asArray();
      if ( array->table() != 0 )
         vm->retval( (int64) array->tablePos() );
      else
         vm->retnil();
      return;
   }

   vm->raiseRTError( new AccessError( ErrorParam( e_prop_acc ) ) );
}


//====================================================//
// THE BOM TABLE
//====================================================//


static void (* const  BOMTable  [] ) ( Falcon::VMachine *) =
{
   BOM_toString,
   BOM_len,
   BOM_first,
   BOM_last,
   BOM_compare,
   BOM_equal,
   BOM_type,
   BOM_className,
   BOM_baseClass,
   BOM_derivedFrom,
   BOM_clone,
   BOM_serialize,
   BOM_attribs,
   BOM_backTrim,
   BOM_frontTrim,
   BOM_allTrim,
   BOM_front,
   BOM_back,
   BOM_table,
   BOM_tabField,
   BOM_tabRow
};

//====================================================
// THE BOM IMPLEMENTATION
//====================================================

bool Item::getBom( const String &property, Item &method, BomMap *bmap ) const
{
   int *value = (int *) bmap->find( &property );
   if ( value == NULL )
      return false;
   method.setFbom( *this, *value );
   return true;
}


bool Item::callBom( VMachine *vm ) const
{
   if( isFbom() )
   {
      // switching here for type may allow to create different BOMs tables for each item type.

      // Switching self/sender. Is it needed?
      Item oldSender = vm->sender();
      vm->sender() = vm->self();

      // real call
      getFbomItem( vm->self() );
      // todo: check validity
      void (* const f)( VMachine *)  = BOMTable[ getFbomMethod() ];
      f( vm );

      // Switching self/sender. Is it needed?
      vm->self() = vm->sender();
      vm->sender() = oldSender;

      return true;
   }

   return false;
}

//====================================================//
// THE BOM UTILITIES
//====================================================//

namespace Fbom {

void toString( VMachine *vm, Item *elem, Item *format )
{

   if ( elem != 0 )
   {

      GarbageString *ret = new GarbageString( vm );

      if ( format != 0 )
      {
         if ( format->isString() )
         {
            Format fmt( *format->asString() );
            if( ! fmt.isValid() )
            {
               vm->raiseRTError( new ParamError( ErrorParam( e_param_fmt_code ) ) );
               return;
            }

            if ( fmt.format( vm, *elem, *ret ) )
            {
               vm->retval( ret );
               return;
            }
         }
         else if ( format->isObject() )
         {
            CoreObject *fmtO = format->asObject();
            if( fmtO->derivedFrom( "Format" ) )
            {
               Format *fmt = static_cast<Format *>( fmtO->getUserData() );
               if ( fmt->format( vm, *elem, *ret ) )
               {
                  vm->retval( ret );
                  return;
               }
            }
         }
      }
      else {
         vm->itemToString( *ret, elem, "" );
         vm->retval( ret );
         return;
      }
   }

   vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "X,[S/O]" ) ) );
}


void makeIterator( VMachine *vm, const Item &self, bool begin )
{
   // create the iterator
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   switch( self.type() )
   {
      case FLC_ITEM_STRING:
      {
         String *orig = self.asString();
         int64 pos = begin ? 0 : (orig->size() == 0 ? 0 : orig->length() - 1);
         iterator->setProperty( "_pos", pos );
      }
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *orig = self.asMemBuf();
         int64 pos = begin ? 0 : (orig->size() == 0 ? 0 : orig->length() - 1);
         iterator->setProperty( "_pos", pos );
      }
      break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *orig = self.asArray();
         int64 pos = begin ? 0 : (orig->length() == 0 ? 0 : orig->length() - 1);
         iterator->setProperty( "_pos", pos );
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *orig = self.asDict();
         DictIterator *iter;
         if( begin )
            iter = orig->first();
         else
            iter = orig->last();
         iterator->setUserData( iter );
      }
      break;

      case FLC_ITEM_ATTRIBUTE:
      {
         Attribute *attrib = self.asAttribute();
         // only from begin.
         iterator->setUserData( attrib->getIterator() );
      }
      break;

      default:
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ) ) );
         return;
   }

   iterator->setProperty( "_origin", self );
   vm->retval( iterator );
}


} // fbom
} // falcon

/* end of fbom.cpp */
