/*
   FALCON - The Falcon Programming Language.
   FILE: core_module.cpp

   Falcon core module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/fstream.h>
#include <falcon/eng_messages.h>
#include <falcon/rampmode.h>

/*#
   @funset core_basic_io Basic I/O
   @brief Functions providing basic interface.

   RTL Basic I/O functions are mainly meant to provide scripts with a
   very basic interface to interact with the outside world.
*/


/*#
   @group core_syssupport System Support
   @brief Function and classes supporting OS and environment.

   This group of functions and classes is meant to provide OS and
   environmental basic support to Falcon scripts.
*/


/*#
   @group memory_manipulation Raw memory manipulation functions.
   @brief DANGEROUS group of functions and methods manipulating memory.

   This group of functions is meant to manipulate directly memory. It is useful
   to support raw bindings from external C modules returning raw memory areas,
   or requiring raw memory areas to be passed to them.

   In Falcon, raw memory pointers are just represented as 64-bit integers, eventually
   cast down to proper sizes on different architectures. In short, they are integers
   that can be directly modified by the Falcon script and fed in the library.

   Messing with functions in this group is the fastest and safest way to find
   troubles. Avoid them unless it is necessary to deal with raw pointers coming
   from outer bindings.
*/

namespace Falcon {

/****************************************
   Module initializer
****************************************/

Module* core_module_init()
{
   Module *self = new Module();
   self->name( "falcon.core" );
   #define FALCON_DECLARE_MODULE self
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( FALCON_VERSION_NUM );

   //=======================================================================
   // Message setting
   //=======================================================================

   StringTable *st = new StringTable(*engineStrings);
   self->adoptStringTable( st, true );
   //=======================================================================
   // Metaclasses
   //=======================================================================
   self->addExtFunc( "len", &Falcon::core::mth_len )->
      addParam("item");
   self->addExtFunc( "toString", &Falcon::core::mth_ToString )->
      addParam("item")->addParam("format");
   self->addExtFunc( "compare", &Falcon::core::mth_compare )->
      addParam("first")->addParam("second");
   self->addExtFunc( "typeId", &Falcon::core::mth_typeId )->
      addParam("item");
   self->addExtFunc( "clone", &Falcon::core::mth_clone )->
      addParam("item");
   self->addExtFunc( "serialize", &Falcon::core::mth_serialize )->
      addParam("item")->addParam("stream");
   self->addExtFunc( "isCallable", &Falcon::core::mth_isCallable )->
      addParam("item");
   self->addExtFunc( "className", &Falcon::core::mth_className )->
      addParam("item");
   self->addExtFunc( "baseClass", &Falcon::core::mth_baseClass )->
      addParam("item");
   self->addExtFunc( "derivedFrom", &Falcon::core::mth_derivedFrom )->
      addParam("item")->addParam("cls");
   self->addExtFunc( "metaclass", &Falcon::core::mth_metaclass )->
      addParam("item");
   self->addExtFunc( "describe", &Falcon::core::mth_describe )->
      addParam("item")->addParam("depth")->addParam("maxLen");
   self->addExtFunc( "isBound", &Falcon::core::mth_bound  )->
      addParam("item");



   /*#
      @group bom_classes Basic Object Model
      @brief Generic item reflection classes

      The classes in this group reflect the the underlying Falcon item types.
      Methods declared by this classes can be applied to every item
      (if they are in the @a BOM class) or to some specific item type.

      The methods can be applied both to symbols containing some data
      of a certain types, or directly to the constants specifying them.
      For example, to obtain the length of the "Hello world" string, it is
      possible to do either:

      @code
         > "Hello world".len()
         // or
         item = "Hello world"
         > item.len()
      @endcode

   */
   /*#
      @class BOM
      @ingroup bom_classes
      @brief Basic Object Model metaclass

      This class contains the methods that can be applied to
      every falcon item; the method themselves are not
      shown in the item list, and the @b provides keyword
      won't detectect their availability unless they have
      been explicitly re-declared (overloaded) by objects,
      instances, classes, array bindings or blessed dictionariees.

      Nevertheless, the method listed in the BOM metaclass can be
      applied to any item, while methods defined in specific
      item metaclasses derived from BOM, as i.e. the @a Dictionary
      metaclass, can be applied only to items of the reflected type.

      @note The method @a BOM.compare is meant to overload the behavior
      of generic VM comparisons, including relational operators
      (<, >, <=, >=, ==, !=) and generic ordering criterions, for example in
      @a Dictionary insertions and @a arraySort.
   */

   /*#
    @method __add BOM
    @brief Overrides binary addition operand.
    @param operand The second operand in the expression.
    @return The value of @b self + @b operand.

    If an object or instance re-defines this method,
    when a "+" operation is performed on this object,
    the method gets called.

    This includes self-target operations as +=, -= and so on; in
    this latter case, the return value of the function will also
    be immediately assigned to this object.

    @note There is no guarantee that the @b operand type
    is the same of this item.
   */

   /*#
    @method __sub BOM
    @brief Overrides binary subtraction operand.
    @param operand The second operand in the expression.
    @return The value of @b self - @b operand.

    If an object or instance re-defines this method,
    when a "-" operation is performed on this object,
    the method gets called.

    This includes self-target operations as +=, -= and so on; in
    this latter case, the return value of the function will also
    be immediately assigned to this object.

    @note There is no guarantee that the @b operand type
    is the same of this item.
   */

   /*#
    @method __mul BOM
    @brief Overrides binary multiplication operand.
    @param operand The second operand in the expression.
    @return The value of @b self * @b operand.

    If an object or instance re-defines this method,
    when a "*" operation is performed on this object,
    the method gets called.

    This includes self-target operations as +=, -= and so on; in
    this latter case, the return value of the function will also
    be immediately assigned to this object.

    @note There is no guarantee that the @b operand type
    is the same of this item.
   */

   /*#
    @method __div BOM
    @brief Overrides binary division operand.
    @param operand The second operand in the expression.
    @return The value of @b self / @b operand.

    If an object or instance re-defines this method,
    when a "/" operation is performed on this object,
    the method gets called.

    This includes self-target operations as +=, -= and so on; in
    this latter case, the return value of the function will also
    be immediately assigned to this object.

    @note There is no guarantee that the @b operand type
    is the same of this item.
   */

   /*#
    @method __mod BOM
    @brief Overrides modulo operand.
    @param operand The second operand in the expression.
    @return The value of @b self % @b operand.

    If an object or instance re-defines this method,
    when a "%" operation is performed on this object,
    the method gets called.

    This includes self-target operations as +=, -= and so on; in
    this latter case, the return value of the function will also
    be immediately assigned to this object.

    @note There is no guarantee that the @b operand type
    is the same of this item.
   */

   /*#
    @method __pow BOM
    @brief Overrides power operand.
    @param operand The second operand in the expression.
    @return The value of @b self ** @b operand.

    If an object or instance re-defines this method,
    when a "**" operation is performed on this object,
    the method gets called.

    This includes self-target operations as +=, -= and so on; in
    this latter case, the return value of the function will also
    be immediately assigned to this object.

    @note There is no guarantee that the @b operand type
    is the same of this item.
   */

   /*#
    @method __inc BOM
    @brief Overrides increment unary prefix operand.
    @param operand The second operand in the expression.
    @return The value of ++ @b self.

    If an object or instance re-defines this method,
    when a "++" prefix operation is performed on this object,
    the method gets called.

    The implementation should modify the object, and return
    itself modified (or value representing this object after
    modification).
   */

   /*#
    @method __dec BOM
    @brief Overrides decrement unary prefix operand.
    @param operand The second operand in the expression.
    @return The value of -- @b self.

    If an object or instance re-defines this method,
    when a "--" prefix operation is performed on this object,
    the method gets called.

    The implementation should modify the object, and return
    itself modified (or value representing this object after
    modification).
   */

   /*#
    @method __incpost BOM
    @brief Overrides increment unary postifx operand.
    @param operand The second operand in the expression.
    @return The value of @b self ++.

    If an object or instance re-defines this method,
    when a "++" postfix operation is performed on this object,
    the method gets called.

    The implementation should keep an unmodified copy of this
    object, modify this instance and return the previous one,
    or a value representing the previous status of this object.
   */

   /*#
    @method __decpost BOM
    @brief Overrides decrement unary postfix operand.
    @param operand The second operand in the expression.
    @return The value of @b self --.

    If an object or instance re-defines this method,
    when a "--" postfix operation is performed on this object,
    the method gets called.

    The implementation should keep an unmodified copy of this
    object, modify this instance and return the previous one,
    or a value representing the previous status of this object.
   */

   /*#
    @method __getIndex BOM
    @brief Overrides array access operator []
    @param index The index of the desired item.
    @return The value of @b self [@b index].

    If an object or instance re-defines this method,
    when a "[]" array access operator is called to read
    a value from a given index, this method gets called.

    The @b index value may be of any type.

    The function should return a value or throw an AccessError
    if the @b index is considered invalid.
   */

   /*#
    @method __setIndex BOM
    @brief Overrides array write operator []
    @param index The index of the desired item.
    @param value The value that must be set.
    @return The value of (@b self [@b index] = @b value).

    If an object or instance re-defines this method,
    when a "[]" array access operator is called to write
    a value to a given index, this method gets called.

    The @b index value may be of any type.

    The function should return the same value that is being
    assigned, but the return value can be freely changed.
   */

   /*#
    @method __call BOM
    @brief Overrides call operator "self()".
    @param ... The parameters passed to the original call.
    @return The value of @b self( ... ).

    If an object or instance re-defines this method,
    when this object is accessed through the call operator,
    then this method is called instead.

    This allows to create functors, as in the following example:

    @code
    object sum
       function __call( a, b )
         return a+b
       end
    end

    > sum( 2, 2 )       // 4
    @endcode
   */

   Falcon::Symbol *bom_meta = self->addClass( "%FBOM" );
   bom_meta->exported( false );
   self->addClassMethod( bom_meta, "len", &Falcon::core::mth_len );
   self->addClassMethod( bom_meta, "toString", &Falcon::core::mth_ToString ).asSymbol()->
      addParam("format");
   self->addClassMethod( bom_meta, "compare", &Falcon::core::mth_compare ).asSymbol()->
      addParam("other");
   self->addClassMethod( bom_meta, "typeId", &Falcon::core::mth_typeId );
   self->addClassMethod( bom_meta, "clone", &Falcon::core::mth_clone );
   self->addClassMethod( bom_meta, "serialize", &Falcon::core::mth_serialize ).asSymbol()->
      addParam("stream");
   self->addClassMethod( bom_meta, "isCallable", &Falcon::core::mth_isCallable );
   self->addClassMethod( bom_meta, "className", &Falcon::core::mth_className );
   self->addClassMethod( bom_meta, "baseClass", &Falcon::core::mth_baseClass );
   self->addClassMethod( bom_meta, "derivedFrom", &Falcon::core::mth_derivedFrom );
   self->addClassMethod( bom_meta, "metaclass", &Falcon::core::mth_metaclass );
   self->addClassMethod( bom_meta, "ptr", &Falcon::core::BOM_ptr );
   self->addClassMethod( bom_meta, "describe", &Falcon::core::mth_describe ).asSymbol()
      ->addParam("depth")->addParam("maxLen");
   self->addClassMethod( bom_meta, "bound", &Falcon::core::mth_bound );

   Falcon::Symbol *nil_meta = self->addClass( "Nil" );
   nil_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   nil_meta->exported( false );
   nil_meta->getClassDef()->setMetaclassFor( FLC_ITEM_NIL );

   Falcon::Symbol *umb_meta = self->addClass( "Unbound" );
   umb_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   umb_meta->exported( false );
   umb_meta->getClassDef()->setMetaclassFor( FLC_ITEM_UNB );

   Falcon::Symbol *bool_meta = self->addClass( "Boolean" );
   bool_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   bool_meta->exported( false );
   bool_meta->getClassDef()->setMetaclassFor( FLC_ITEM_BOOL );

   /*#
      @class Integer
      @from BOM
      @ingroup bom_classes
      @brief Integer type basic object model metaclass.
   */
   Falcon::Symbol *int_meta = self->addClass( "Integer" );
   int_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   int_meta->exported( false );
   int_meta->getClassDef()->setMetaclassFor( FLC_ITEM_INT );
   self->addClassMethod( int_meta, "times", &Falcon::core::core_times ).asSymbol()->setEta( true )->
      addParam("count")->addParam("sequence");
   self->addClassMethod( int_meta, "upto", &Falcon::core::core_upto ).asSymbol()->setEta( true )->
      addParam("limit")->addParam("sequence");
   self->addClassMethod( int_meta, "downto", &Falcon::core::core_downto ).asSymbol()->setEta( true )->
      addParam("limit")->addParam("sequence");
   self->addClassMethod( int_meta, "ptr", &Falcon::core::Integer_ptr );

   /*#
      @class Numeric
      @from BOM
      @ingroup bom_classes
      @brief Generic number type basic object model metaclass.
   */
   Falcon::Symbol *num_meta = self->addClass( "Numeric" );
   num_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   num_meta->exported( false );
   num_meta->getClassDef()->setMetaclassFor( FLC_ITEM_NUM );

   /*#
      @class Range
      @from BOM
      @ingroup bom_classes
      @brief Metaclass for Falcon range type.

      This class holds the methods that can be applied to Falcon range items.
      Ranges are created through the @b [:] operator, like this:
      @code
         r1 = [1:10]          // 1 to 10
         r2 = [-1:0]          // reverse sequence
         r3 = [0:10:2]        // stepping range

         > "Hello world"[r2]  // reverses the string
      @endcode
   */
   Falcon::Symbol *range_meta = self->addClass( "Range" );
   range_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   range_meta->exported( false );
   range_meta->getClassDef()->setMetaclassFor( FLC_ITEM_RANGE );
   self->addClassMethod( range_meta, "times", &Falcon::core::core_times ).asSymbol()->setEta( true )->
      addParam("count")->addParam("sequence");

   /*#
      @class LateBinding
      @from BOM
      @ingroup bom_classes
      @brief Metaclass for LateBinding type.

      Late bindings are special items treated as "symbols", that
      can receive an associated value at runtime.
   */
   Falcon::Symbol *lbind_meta = self->addClass( "LateBinding" );
   lbind_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   lbind_meta->exported( false );
   lbind_meta->getClassDef()->setMetaclassFor( FLC_ITEM_LBIND );
   self->addClassMethod( lbind_meta, "value", &Falcon::core::LateBinding_value );
   self->addClassMethod( lbind_meta, "bound", &Falcon::core::LateBinding_bound );
   self->addClassMethod( lbind_meta, "bind", &Falcon::core::LateBinding_bind );
   self->addClassMethod( lbind_meta, "unbind", &Falcon::core::LateBinding_unbind );

   /*#
      @class Function
      @from BOM
      @ingroup bom_classes
      @brief Metaclass for Falcon function type.
   */
   Falcon::Symbol *func_meta = self->addClass( "Function" );
   func_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   func_meta->exported( false );
   func_meta->getClassDef()->setMetaclassFor( FLC_ITEM_FUNC );
   self->addClassMethod( func_meta, "attributes", &Falcon::core::Function_attributes );
   self->addClassMethod( func_meta, "name", &Falcon::core::Function_name );
   self->addClassMethod( func_meta, "caller", &Falcon::core::Function_caller ).asSymbol()->
      addParam("level");    //static
   self->addClassMethod( func_meta, "trace", &Falcon::core::Function_trace ).asSymbol()->
      addParam("level");    //static

   /*#
      @class GarbagePointer
      @from BOM
      @ingroup bom_classes
      @brief Metaclass for internal and application-wide garbage sensible data.

      GarbagePointer is a class used to reflect opaque data which can be automatically
      destroyed by the Falcon Garbage Collector when it goes out of scope.
   */
   Falcon::Symbol *gcptr_meta = self->addClass( "GarbagePointer" );
   gcptr_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   gcptr_meta->exported( false );
   gcptr_meta->getClassDef()->setMetaclassFor( FLC_ITEM_GCPTR );
   self->addClassMethod( gcptr_meta, "ptr", &Falcon::core::GarbagePointer_ptr );

   //==================================================================
   // String class
   //
   self->addExtFunc( "strFront", &Falcon::core::mth_strFront )->
      addParam("str")->addParam("count")->addParam("remove")->addParam("numeric");
   self->addExtFunc( "strBack", &Falcon::core::mth_strBack )->
      addParam("str")->addParam("count")->addParam("remove")->addParam("numeric");
   self->addExtFunc( "strFrontTrim", &Falcon::core::mth_strFrontTrim )->
      addParam("str")->addParam("trimSet");
   self->addExtFunc( "strBackTrim", &Falcon::core::mth_strBackTrim )->
      addParam("str")->addParam("trimSet");
   self->addExtFunc( "strTrim", &Falcon::core::mth_strTrim )->
      addParam("str")->addParam("trimSet");

   /*#
      @class String
      @from BOM
      @ingroup bom_classes
      @brief Metaclass for string items.

      This is the set of methods that can be applied to string items.
   */

   Falcon::Symbol *string_meta = self->addClass( "String" );
   string_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   string_meta->exported( false );
   string_meta->getClassDef()->setMetaclassFor( FLC_ITEM_STRING );
   self->addClassMethod( string_meta, "front", &Falcon::core::mth_strFront ).asSymbol()->
      addParam("count")->addParam("remove")->addParam("numeric");
   self->addClassMethod( string_meta, "fill", &Falcon::core::mth_strFill ).asSymbol()->
      addParam("chr");
   self->addClassMethod( string_meta, "back", &Falcon::core::mth_strBack ).asSymbol()->
      addParam("count")->addParam("remove")->addParam("numeric");
   self->addClassMethod( string_meta, "ftrim", &Falcon::core::mth_strFrontTrim ).asSymbol()->
      addParam("trimSet");
   self->addClassMethod( string_meta, "rtrim", &Falcon::core::mth_strBackTrim ).asSymbol()->
      addParam("trimSet");
   self->addClassMethod( string_meta, "trim", &Falcon::core::mth_strTrim ).asSymbol()->
      addParam("trimSet");
   //self->addClassMethod( string_meta, "first", &Falcon::core::String_first );
   //self->addClassMethod( string_meta, "last", &Falcon::core::String_last );
   self->addClassMethod( string_meta, "split", &Falcon::core::mth_strSplit ).asSymbol()
      ->addParam( "token" )->addParam( "count" );
   self->addClassMethod( string_meta, "splittr", &Falcon::core::mth_strSplitTrimmed ).asSymbol()
      ->addParam( "token" )->addParam("count");
   self->addClassMethod( string_meta, "merge", &Falcon::core::mth_strMerge ).asSymbol()
      ->addParam("array")->addParam("mergeStr")->addParam("count");
   self->addClassMethod( string_meta, "join", &Falcon::core::String_join );
   self->addClassMethod( string_meta, "find", &Falcon::core::mth_strFind ).asSymbol()
      ->addParam("needle")->addParam("start")->addParam("end");
   self->addClassMethod( string_meta, "rfind", &Falcon::core::mth_strBackFind ).asSymbol()
      ->addParam("needle")->addParam("start")->addParam("end");
   self->addClassMethod( string_meta, "replace", &Falcon::core::mth_strReplace ).asSymbol()
      ->addParam("substr")->addParam("repstr")->addParam("start")->addParam("end");
   self->addClassMethod( string_meta,"replicate", &Falcon::core::mth_strReplicate ).asSymbol()
      ->addParam("times");
   self->addClassMethod( string_meta,"esq", &Falcon::core::mth_strEsq ).asSymbol()
      ->addParam("inplace");
   self->addClassMethod( string_meta,"unesq", &Falcon::core::mth_strUnesq ).asSymbol()
      ->addParam("inplace");
   self->addClassMethod( string_meta,"escape", &Falcon::core::mth_strEscape ).asSymbol()
      ->addParam("full");
   self->addClassMethod( string_meta,"unescape", &Falcon::core::mth_strUnescape ).asSymbol()
      ->addParam("inplace");
   self->addClassMethod( string_meta, "upper", &Falcon::core::mth_strUpper );
   self->addClassMethod( string_meta, "lower", &Falcon::core::mth_strLower );
   self->addClassMethod( string_meta, "cmpi", &Falcon::core::mth_strCmpIgnoreCase ).asSymbol()
      ->addParam("string2");
   self->addClassMethod( string_meta, "wmatch", &Falcon::core::mth_strWildcardMatch ).asSymbol()
      ->addParam("wildcard")->addParam("ignoreCase");
   self->addClassMethod( string_meta, "toMemBuf", &Falcon::core::mth_strToMemBuf ).asSymbol()
      ->addParam("wordWidth");
   self->addClassMethod( string_meta, "ptr", &Falcon::core::String_ptr );
   self->addClassMethod( string_meta, "charSize", &Falcon::core::String_charSize ).asSymbol()
      ->addParam("bpc");
   self->addClassMethod( string_meta, "startsWith", &Falcon::core::mth_strStartsWith ).asSymbol()
      ->addParam("token")->addParam("icase");
   self->addClassMethod( string_meta, "endsWith", &Falcon::core::mth_strEndsWith ).asSymbol()
      ->addParam("token")->addParam("icase");

   //==================================================================
   // Array class
   //
   Falcon::Symbol *array_meta = self->addClass( "Array" );
   array_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   array_meta->exported( false );
   array_meta->getClassDef()->setMetaclassFor( FLC_ITEM_ARRAY );

   /*#
      @class Array
      @from BOM
      @ingroup bom_classes
      @brief MetaClass for Falcon arrays.

      This is the class reflecting the base array classes.
   */
   self->addClassMethod( array_meta, "front", &Falcon::core::Array_front ).asSymbol()->
      addParam("remove");
   self->addClassMethod( array_meta, "back", &Falcon::core::Array_back ).asSymbol()->
      addParam("remove");
   self->addClassMethod( array_meta, "first", &Falcon::core::Array_first );
   self->addClassMethod( array_meta, "last", &Falcon::core::Array_last );
   self->addClassMethod( array_meta, "table", &Falcon::core::Array_table );
   self->addClassMethod( array_meta, "tabField", &Falcon::core::Array_tabField ).asSymbol()->
      addParam("field");
   self->addClassMethod( array_meta, "tabRow", &Falcon::core::Array_tabRow );
   self->addClassMethod( array_meta, "ins", &Falcon::core::mth_arrayIns ).asSymbol()
      ->addParam("itempos")->addParam("item");
   self->addClassMethod( array_meta, "del", &Falcon::core::mth_arrayDel ).asSymbol()
      ->addParam("item");
   self->addClassMethod( array_meta, "delAll", &Falcon::core::mth_arrayDelAll ).asSymbol()
      ->addParam("item");
   self->addClassMethod( array_meta, "add", &Falcon::core::mth_arrayAdd ).asSymbol()
      ->addParam("item");
   self->addClassMethod( array_meta, "resize", &Falcon::core::mth_arrayResize ).asSymbol()->
      addParam("newSize");
   self->addClassMethod( array_meta, "find", &Falcon::core::mth_arrayFind ).asSymbol()->
      addParam("item")->addParam("start")->addParam("end");
   self->addClassMethod( array_meta, "scan", &Falcon::core::mth_arrayScan ).asSymbol()->
      addParam("func")->addParam("start")->addParam("end");
   self->addClassMethod( array_meta, "sort", &Falcon::core::mth_arraySort ).asSymbol()->
      addParam("sortingFunc");
   self->addClassMethod( array_meta, "remove", &Falcon::core::mth_arrayRemove ).asSymbol()->
      addParam("itemPos")->addParam("lastItemPos");
   self->addClassMethod( array_meta, "merge", &Falcon::core::mth_arrayMerge ).asSymbol()->
      addParam("array");
   self->addClassMethod( array_meta, "fill", &Falcon::core::mth_arrayFill ).asSymbol()->
      addParam("item");
   self->addClassMethod( array_meta, "head", &Falcon::core::mth_arrayHead );
   self->addClassMethod( array_meta, "tail", &Falcon::core::mth_arrayTail );

   self->addClassMethod( array_meta, "getProperty", &Falcon::core::mth_getProperty ).asSymbol()->
      addParam("propName");
   self->addClassMethod( array_meta, "setProperty", &Falcon::core::mth_setProperty ).asSymbol()->
      addParam("propName")->addParam("value");
   self->addClassMethod( array_meta, "properties", &Falcon::core::mth_properties );
   self->addClassMethod( array_meta, "comp", &Falcon::core::Array_comp ).asSymbol()->
      addParam("source")->addParam("filter");
   self->addClassMethod( array_meta, "mcomp", &Falcon::core::Array_mcomp );
   self->addClassMethod( array_meta, "mfcomp", &Falcon::core::Array_mfcomp ).asSymbol()->
         addParam("filter");
   
   self->addClassMethod( array_meta, "compact", &Falcon::core::mth_arrayCompact );
   self->addClassMethod( array_meta, "NM", &Falcon::core::mth_arrayNM );

   //==================================================================
   // Dict class
   //
   self->addExtFunc( "dictFront", &Falcon::core::mth_dictFront )->
      addParam("dict")->addParam("remove")->addParam("key");
   self->addExtFunc( "dictBack", &Falcon::core::mth_dictBack )->
      addParam("dict")->addParam("remove")->addParam("key");

   Falcon::Symbol *dict_meta = self->addClass( "Dictionary" );
   dict_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   dict_meta->exported( false );
   dict_meta->getClassDef()->setMetaclassFor( FLC_ITEM_DICT );

   /*#
      @class Dictionary
      @from BOM
      @ingroup bom_classes
      @brief Metaclass for Falcon dictionary types.

      This class holds the methods that can be applied to Falcon dictionary items.
   */
   self->addClassMethod( dict_meta, "front", &Falcon::core::mth_dictFront ).asSymbol()->
      addParam("remove")->addParam("key");
   self->addClassMethod( dict_meta, "back", &Falcon::core::mth_dictBack ).asSymbol()->
      addParam("remove")->addParam("key");
   self->addClassMethod( dict_meta, "first", &Falcon::core::Dictionary_first );
   self->addClassMethod( dict_meta, "last", &Falcon::core::Dictionary_last );
   self->addClassMethod( dict_meta, "comp", &Falcon::core::Dictionary_comp ).asSymbol()->
      addParam( "source" )->addParam( "filter" );
   self->addClassMethod( dict_meta, "mcomp", &Falcon::core::Dictionary_mcomp );
   self->addClassMethod( dict_meta, "mfcomp", &Falcon::core::Dictionary_mfcomp ).asSymbol()->
      addParam( "filter" );


   self->addClassMethod( dict_meta, "merge", &Falcon::core::mth_dictMerge ).asSymbol()->
      addParam("sourceDict");
   self->addClassMethod( dict_meta, "keys", &Falcon::core::mth_dictKeys );
   self->addClassMethod( dict_meta,  "values", &Falcon::core::mth_dictValues );
   self->addClassMethod( dict_meta,  "get", &Falcon::core::mth_dictGet ).asSymbol()->
      addParam("key");
   self->addClassMethod( dict_meta,  "set", &Falcon::core::mth_dictSet ).asSymbol()->
      addParam("key")->addParam("value");
   self->addClassMethod( dict_meta, "find", &Falcon::core::mth_dictFind ).asSymbol()->
      addParam("key");
   self->addClassMethod( dict_meta, "best", &Falcon::core::mth_dictBest ).asSymbol()->
      addParam("key");
   self->addClassMethod( dict_meta, "remove", &Falcon::core::mth_dictRemove ).asSymbol()->
      addParam("key");
   self->addClassMethod( dict_meta, "fill", &Falcon::core::mth_dictFill ).asSymbol()->
      addParam("item");
   self->addClassMethod( dict_meta, "clear", &Falcon::core::mth_dictClear );

   self->addClassMethod( dict_meta, "getProperty", &Falcon::core::mth_getProperty ).asSymbol()->
      addParam("propName");
   self->addClassMethod( dict_meta, "setProperty", &Falcon::core::mth_setProperty ).asSymbol()->
      addParam("propName")->addParam("value");
   self->addClassMethod( dict_meta, "properties", &Falcon::core::mth_properties );
   
   self->addClassMethod( dict_meta, "dop", &Falcon::core::Dictionary_dop ).asSymbol()->
      addParam("key")->addParam("dflt")->addParam("orig");

   //==================================================================
   // Object class
   //
   /*#
      @class Object
      @from BOM
      @ingroup bom_classes
      @brief Object (class instance) type basic object model metaclass.
   */
   Falcon::Symbol *object_meta = self->addClass( "Object" );
   object_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   object_meta->exported( false );
   object_meta->getClassDef()->setMetaclassFor( FLC_ITEM_OBJECT );
   self->addClassMethod( object_meta, "attributes", &Falcon::core::Object_attributes );
   self->addClassMethod( object_meta, "getProperty", &Falcon::core::mth_getProperty ).asSymbol()->
      addParam("propName");
   self->addClassMethod( object_meta, "setProperty", &Falcon::core::mth_setProperty ).asSymbol()->
      addParam("propName")->addParam("value");
   self->addClassMethod( object_meta, "properties", &Falcon::core::mth_properties );
   self->addClassMethod( object_meta, "comp", &Falcon::core::Object_comp ).asSymbol()->
         addParam("source")->addParam("filter");
   self->addClassMethod( object_meta, "mcomp", &Falcon::core::Object_mcomp );
   self->addClassMethod( object_meta, "mfcomp", &Falcon::core::Object_mfcomp ).asSymbol()->
         addParam("filter");
   self->addClassMethod( object_meta, "apply", &Falcon::core::Object_apply );
   self->addClassMethod( object_meta, "getState", &Falcon::core::Object_getState );
   self->addClassMethod( object_meta, "setState", &Falcon::core::Object_setState );
   self->addClassMethod( object_meta, "retrieve", &Falcon::core::Object_retrieve );

   //==================================================================
   // MemoryBuffer class
   //
   /* Docs for this class are in membuf_ext.cpp */
   Falcon::Symbol *membuf_meta = self->addClass( "MemoryBuffer" );
   membuf_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   membuf_meta->exported( false );
   membuf_meta->getClassDef()->setMetaclassFor( FLC_ITEM_MEMBUF );
   self->addClassMethod( membuf_meta, "front", &Falcon::core::MemoryBuffer_front );
   self->addClassMethod( membuf_meta, "back", &Falcon::core::MemoryBuffer_back );
   self->addClassMethod( membuf_meta, "first", &Falcon::core::MemoryBuffer_first );
   self->addClassMethod( membuf_meta, "last", &Falcon::core::MemoryBuffer_last );

   self->addClassMethod( membuf_meta, "put", &Falcon::core::MemoryBuffer_put ).asSymbol()->
      addParam( "data" );
   self->addClassMethod( membuf_meta, "get", &Falcon::core::MemoryBuffer_get );
   self->addClassMethod( membuf_meta, "rewind", &Falcon::core::MemoryBuffer_rewind );
   self->addClassMethod( membuf_meta, "reset", &Falcon::core::MemoryBuffer_reset );
   self->addClassMethod( membuf_meta, "flip", &Falcon::core::MemoryBuffer_flip );
   self->addClassMethod( membuf_meta, "limit", &Falcon::core::MemoryBuffer_limit ).asSymbol()->
      addParam( "pos" );
   self->addClassMethod( membuf_meta, "mark", &Falcon::core::MemoryBuffer_mark );
   self->addClassMethod( membuf_meta, "position", &Falcon::core::MemoryBuffer_position ).asSymbol()->
      addParam( "pos" );
   self->addClassMethod( membuf_meta, "clear", &Falcon::core::MemoryBuffer_clear );
   self->addClassMethod( membuf_meta, "compact", &Falcon::core::MemoryBuffer_compact );
   self->addClassMethod( membuf_meta, "remaining", &Falcon::core::MemoryBuffer_remaining );
   self->addClassMethod( membuf_meta, "fill", &Falcon::core::MemoryBuffer_fill ).asSymbol()->
      addParam("value");
   self->addClassMethod( membuf_meta, "wordSize", &Falcon::core::MemoryBuffer_wordSize );
   self->addClassMethod( membuf_meta, "ptr", &Falcon::core::MemoryBuffer_ptr );


   /*#
      @class ClassMethod
      @from BOM
      @ingroup bom_classes
      @brief Metaclass of ClassMethod items.

      ClassMethods are special methods that, once applied to objects or
      methods, resolve in a new method taking the original object, but the
      function/code of one of the base classes.
   */
   Falcon::Symbol *clsmethod_meta = self->addClass( "ClassMethod" );
   clsmethod_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   clsmethod_meta->exported( false );
   clsmethod_meta->getClassDef()->setMetaclassFor( FLC_ITEM_CLSMETHOD );
   self->addClassMethod( clsmethod_meta, "attributes", &Falcon::core::Method_attributes );

   /*#
      @class Method
      @from BOM
      @ingroup bom_classes
      @brief Metaclass of method items.

      This is the class reflecting falcon method items. A method is a set of
      an item and a function applied to that. The item can be a method again,
      as it is possible to apply methods to method items.
   */
   Falcon::Symbol *method_meta = self->addClass( "Method" );
   method_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   method_meta->exported( false );
   method_meta->getClassDef()->setMetaclassFor( FLC_ITEM_METHOD );
   self->addClassMethod( method_meta, "source", &Falcon::core::Method_source );
   self->addClassMethod( method_meta, "base", &Falcon::core::Method_base );
   self->addClassMethod( method_meta, "attributes", &Falcon::core::Method_attributes );

   /*#
      @class Class
      @from BOM
      @ingroup bom_classes
      @brief Metaclass of Class items.

      This is the class reflecting falcon class items. A class is a callable
      item that can generate instances.
   */
   Falcon::Symbol *class_meta = self->addClass( "Class" );
   class_meta->getClassDef()->addInheritance( new Falcon::InheritDef( bom_meta ) );
   class_meta->exported( false );
   class_meta->getClassDef()->setMetaclassFor( FLC_ITEM_CLASS );
   self->addClassMethod( class_meta, "attributes", &Falcon::core::Class_attributes );
   self->addClassMethod( class_meta, "properties", &Falcon::core::mth_properties );

   //=======================================================================
   // Module declaration body
   //=======================================================================

   /*#
      @global args
      @brief Script arguments
      @ingroup general_purpose

      A global variable holding an array that contains the strings passed as argument for
      the script. Embedders may change the convention, and pass any Falcon item as
      arguments; however, falcon command line and the other standard tools pass only
      an array of strings.
   */
   self->addGlobal( "args", true );

   /*#
      @global scriptName
      @brief Logical module name of current module
      @ingroup general_purpose

      It's a global variable that is usually filled with the script name. It's the logical
      script name that the VM has assigned to this module, mainly used for debugging.
   */
   self->addGlobal( "scriptName", true );

   /*#
      @global scriptPath
      @brief Complete path used to load the script
      @ingroup general_purpose

      It's a global variable that is usually filled with the location from which the script
      has been loaded. It's semantic may vary among embedding applications, but it should
      usually receive the complete path to the main script, in Falcon file convention
      (forward slashes to separate directories), or the complete URI where applicable.
   */
   self->addGlobal( "scriptPath", true );


   self->addExtFunc( "attributes", &Falcon::core::attributes );
   self->addExtFunc( "argv", &Falcon::core::core_argv );
   self->addExtFunc( "argd", &Falcon::core::core_argd );
   self->addExtFunc( "passvp", &Falcon::core::core_passvp );

   self->addExtFunc( "chr", &Falcon::core::chr )->
      addParam("number");
   self->addExtFunc( "ord", &Falcon::core::ord )->
      addParam("string");


   self->addExtFunc( "getProperty", &Falcon::core::mth_getProperty )->
      addParam("obj")->addParam("propName");
   self->addExtFunc( "setProperty", &Falcon::core::mth_setProperty )->
      addParam("obj")->addParam("propName")->addParam("value");
   self->addExtFunc( "properties", &Falcon::core::mth_properties )->
      addParam("obj");

   self->addExtFunc( "yield", &Falcon::core::yield );
   self->addExtFunc( "yieldOut", &Falcon::core::yieldOut )->
      addParam("retval");
   self->addExtFunc( "sleep", &Falcon::core::_f_sleep )->
      addParam("time");
   self->addExtFunc( "beginCritical", &Falcon::core::beginCritical );
   self->addExtFunc( "endCritical", &Falcon::core::endCritical );

   self->addExtFunc( "int", &Falcon::core::val_int )->
      addParam("item");
   self->addExtFunc( "numeric", &Falcon::core::val_numeric )->
      addParam("item");
   self->addExtFunc( "typeOf", &Falcon::core::mth_typeId )->
      addParam("item");
   self->addExtFunc( "exit", &Falcon::core::core_exit )->
      addParam("value");

   self->addExtFunc( "paramCount", &Falcon::core::paramCount );
   self->addExtFunc( "paramNumber", &Falcon::core::_parameter );
   self->addExtFunc( "parameter", &Falcon::core::_parameter )->
      addParam("pnum");
   self->addExtFunc( "paramIsRef", &Falcon::core::paramIsRef )->
      addParam("number");
   self->addExtFunc( "paramSet", &Falcon::core::paramSet )->
      addParam("number")->addParam("value");
   self->addExtFunc( "PageDict", &Falcon::core::PageDict )->
      addParam("pageSize");
   self->addExtFunc( "MemBuf", &Falcon::core::Make_MemBuf )->
      addParam("size")->addParam("wordSize");
   self->addExtFunc( "MemBufFromPtr", &Falcon::core::Make_MemBufFromPtr )->
      addParam("data")->addParam("size")->addParam("wordSize");

   // Creating the TraceStep class:
   // ... first the constructor
   /*Symbol *ts_init = self->addExtFunc( "TraceStep._init", &Falcon::core::TraceStep_init );

   //... then the class
   Symbol *ts_class = self->addClass( "TraceStep", ts_init );

   // then add var props; flc_CLSYM_VAR is 0 and is linked correctly by the VM.
   self->addClassProperty( ts_class, "module" );
   self->addClassProperty( ts_class, "symbol" );
   self->addClassProperty( ts_class, "pc" );
   self->addClassProperty( ts_class, "line" );
   // ... finally add a method, using the symbol that this module returns.
   self->addClassMethod( ts_class, "toString",
      self->addExtFunc( "TraceStep.toString", &Falcon::core::TraceStep_toString ) );*/

   // Creating the Error class class
   Symbol *error_init = self->addExtFunc( "Error._init", &Falcon::core::Error_init );
   Symbol *error_class = self->addClass( "Error", error_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   error_class->getClassDef()->factory( &Falcon::core::ErrorObjectFactory );
   error_class->setWKS( true );

   self->addClassMethod( error_class, "toString",
         self->addExtFunc( "Error.toString", &Falcon::core::Error_toString ) ).setReadOnly( true );
   self->addClassMethod( error_class, "heading", &Falcon::core::Error_heading ).setReadOnly( true );

   // separated property description to test for separate @property faldoc command
   /*#
      @property code Error
      @brief Error code associated with this error.
   */
   self->addClassProperty( error_class, "code" ).
      setReflectFunc( Falcon::core::Error_code_rfrom, &Falcon::core::Error_code_rto );
   self->addClassProperty( error_class, "description" ).
      setReflectFunc( Falcon::core::Error_description_rfrom, &Falcon::core::Error_description_rto );
   self->addClassProperty( error_class, "message" ).
      setReflectFunc( Falcon::core::Error_message_rfrom, &Falcon::core::Error_message_rto );
   self->addClassProperty( error_class, "systemError" ).
      setReflectFunc( Falcon::core::Error_systemError_rfrom, &Falcon::core::Error_systemError_rto );

   /*#
       @property origin Error
       @brief String identifying the origin of the error.

      This code allows to determine  what element of the Falcon engine has raised the error
      (or eventually, if this error has been raised by a script or a loaded module).

      The error origin is a string; when an error gets displayed through a standard
      rendering function (as the Error.toString() method), it is indicated by two
      letters in front of the error code for better readability. The origin code may
      be one of the following:

      - @b compiler - (represented in Error.toString() as CO)
      - @b assembler - (AS)
      - @b loader -  that is, the module loader (LD)
      - @b vm - the virtual machine (when not running a script, short VM)
      - @b script - (that is, a VM running a script, short SS)
      - @b runtime - (core or runtime modules, RT)
      - @b module - an extension module (MD).
      -
   */

   self->addClassProperty( error_class, "origin" ).
         setReflectFunc( Falcon::core::Error_origin_rfrom, &Falcon::core::Error_origin_rto );
   self->addClassProperty( error_class, "module" ).
         setReflectFunc( Falcon::core::Error_module_rfrom, &Falcon::core::Error_module_rto );
   self->addClassProperty( error_class, "symbol" ).
         setReflectFunc( Falcon::core::Error_symbol_rfrom, &Falcon::core::Error_symbol_rto );
   self->addClassProperty( error_class, "line" ).
         setReflectFunc( Falcon::core::Error_line_rfrom, &Falcon::core::Error_line_rto );
   self->addClassProperty( error_class, "pc" ).
         setReflectFunc( Falcon::core::Error_pc_rfrom, &Falcon::core::Error_pc_rto );
   self->addClassProperty( error_class, "subErrors" ).
         setReflectFunc( Falcon::core::Error_subErrors_rfrom );
   self->addClassMethod( error_class, "getSysErrorDesc", &Falcon::core::Error_getSysErrDesc ).setReadOnly( true );

   // Other derived error classes.
   Falcon::Symbol *synerr_cls = self->addClass( "SyntaxError", &Falcon::core::SyntaxError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   synerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   synerr_cls->setWKS( true );

   Falcon::Symbol *genericerr_cls = self->addClass( "GenericError", &Falcon::core::GenericError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   genericerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   genericerr_cls->setWKS( true );

   Falcon::Symbol *codeerr_cls = self->addClass( "CodeError", &Falcon::core::CodeError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   codeerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   codeerr_cls->setWKS( true );

   Falcon::Symbol *rangeerr_cls = self->addClass( "AccessError", &Falcon::core::AccessError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   rangeerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   rangeerr_cls->setWKS( true );

   Falcon::Symbol *matherr_cls = self->addClass( "MathError", &Falcon::core::MathError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   matherr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   matherr_cls->setWKS( true );

   Falcon::Symbol *ioerr_cls = self->addClass( "IoError", &Falcon::core::IoError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   ioerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   ioerr_cls->setWKS( true );

   Falcon::Symbol *typeerr_cls = self->addClass( "TypeError", &Falcon::core::TypeError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   typeerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   typeerr_cls->setWKS( true );

   Falcon::Symbol *paramerr_cls = self->addClass( "ParamError", &Falcon::core::ParamError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   paramerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   paramerr_cls->setWKS( true );

   Falcon::Symbol *parsererr_cls = self->addClass( "ParseError", &Falcon::core::ParseError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   parsererr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   parsererr_cls->setWKS( true );

   Falcon::Symbol *cloneerr_cls = self->addClass( "CloneError", &Falcon::core::CloneError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   cloneerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   cloneerr_cls->setWKS( true );

   Falcon::Symbol *interr_cls = self->addClass( "InterruptedError", &Falcon::core::IntrruptedError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   interr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   interr_cls->setWKS( true );

   Falcon::Symbol *msgerr_cls = self->addClass( "MessageError", &Falcon::core::MessageError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   msgerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   msgerr_cls->setWKS( true );

   Falcon::Symbol *tableerr_cls = self->addClass( "TableError", &Falcon::core::TableError_init )
      ->addParam( "code" )->addParam( "description")->addParam( "extra" );
   tableerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
   tableerr_cls->setWKS( true );

   //=========================================

   // Creating the semaphore class -- will be a FalconObject
   Symbol *semaphore_init = self->addExtFunc( "Semaphore._init", &Falcon::core::Semaphore_init );
   Symbol *semaphore_class = self->addClass( "Semaphore", semaphore_init );

   self->addClassMethod( semaphore_class, "post", &Falcon::core::Semaphore_post ).asSymbol()->
      addParam("count");
   self->addClassMethod( semaphore_class, "wait", &Falcon::core::Semaphore_wait ).asSymbol()->
      addParam("timeout");

   // GC support
   Symbol *gcsing = self->addSingleton( "GC" );
   Symbol *gc_cls = gcsing->getInstance();
   gc_cls->getClassDef()->factory(&Falcon::core::GC_Factory);
   self->addClassProperty( gc_cls, "usedMem" ).
      setReflectFunc( &Falcon::core::GC_usedMem_rfrom );
   self->addClassProperty( gc_cls, "aliveMem" ).
      setReflectFunc( &Falcon::core::GC_aliveMem_rfrom );
   self->addClassProperty( gc_cls, "items" ).
      setReflectFunc( &Falcon::core::GC_items_rfrom );
   self->addClassProperty( gc_cls, "th_normal" ).
      setReflectFunc( &Falcon::core::GC_th_normal_rfrom, &Falcon::core::GC_th_normal_rto );
   self->addClassProperty( gc_cls, "th_active" ).
      setReflectFunc( &Falcon::core::GC_th_active_rfrom, &Falcon::core::GC_th_active_rto );
   self->addClassMethod( gc_cls, "enable", &Falcon::core::GC_enable ).setReadOnly(true).asSymbol()->
      addParam("mode");
   self->addClassMethod( gc_cls, "perform", &Falcon::core::GC_perform ).setReadOnly(true).asSymbol()->
      addParam("wcoll");
   self->addClassMethod( gc_cls, "adjust", &Falcon::core::GC_adjust ).setReadOnly(true).asSymbol()->
      addParam("mode");
   self->addClassProperty( gc_cls, "ADJ_NONE" ).setInteger(RAMP_MODE_OFF).setReadOnly(true);
   self->addClassProperty( gc_cls, "ADJ_STRICT" ).setInteger(RAMP_MODE_STRICT_ID).setReadOnly(true);
   self->addClassProperty( gc_cls, "ADJ_LOOSE" ).setInteger(RAMP_MODE_LOOSE_ID).setReadOnly(true);
   self->addClassProperty( gc_cls, "ADJ_SMOOTH_SLOW" ).setInteger(RAMP_MODE_SMOOTH_SLOW_ID).setReadOnly(true);
   self->addClassProperty( gc_cls, "ADJ_SMOOTH_FAST" ).setInteger(RAMP_MODE_SMOOTH_FAST_ID).setReadOnly(true);

   // VM support
   self->addExtFunc( "vmVersionInfo", &Falcon::core::vmVersionInfo );
   self->addExtFunc( "vmVersionName", &Falcon::core::vmVersionName );
   self->addExtFunc( "vmSystemType", &Falcon::core::vmSystemType );
   self->addExtFunc( "vmModuleVersionInfo", &Falcon::core::vmModuleVersionInfo );
   self->addExtFunc( "vmIsMain", &Falcon::core::vmIsMain );
   self->addExtFunc( "vmFalconPath", &Falcon::core::vmFalconPath );
   self->addExtFunc( "vmSearchPath", &Falcon::core::vmSearchPath );
   self->addExtFunc( "vmModuleName", &Falcon::core::vmModuleName );
   self->addExtFunc( "vmModulePath", &Falcon::core::vmModulePath );
   self->addExtFunc( "vmRelativePath", &Falcon::core::vmRelativePath );

   // Format
   Symbol *format_class = self->addClass( "Format", &Falcon::core::Format_init );
   //format_class->getClassDef()->setObjectManager( &core_falcon_data_manager );
   self->addClassMethod( format_class, "format", &Falcon::core::Format_format ).asSymbol()->
      addParam("item")->addParam("dest");
   self->addClassMethod( format_class, "parse", &Falcon::core::Format_parse ).asSymbol()->
      addParam("fmtspec");
   self->addClassMethod( format_class, "toString", &Falcon::core::Format_toString );
   self->addClassProperty( format_class,"size" );
   self->addClassProperty( format_class, "decimals" );
   self->addClassProperty( format_class, "paddingChr" );
   self->addClassProperty( format_class, "groupingChr" );
   self->addClassProperty( format_class, "decimalChr" );
   self->addClassProperty( format_class, "grouiping" );
   self->addClassProperty( format_class, "fixedSize" );
   self->addClassProperty( format_class, "rightAlign" );
   self->addClassProperty( format_class, "originalFormat" );
   self->addClassProperty( format_class, "misAct" );
   self->addClassProperty( format_class, "convType" );
   self->addClassProperty( format_class, "nilFormat" );
   self->addClassProperty( format_class, "negFormat" );
   self->addClassProperty( format_class, "numFormat" );

   // Iterators
   Symbol *iterator_class = self->addClass( "Iterator", &Falcon::core::Iterator_init );
   iterator_class->setWKS( true );
   iterator_class->addParam("collection")->addParam( "atEnd" );
   self->addClassMethod( iterator_class, "hasCurrent", &Falcon::core::Iterator_hasCurrent );
   self->addClassMethod( iterator_class, "hasNext", &Falcon::core::Iterator_hasNext );
   self->addClassMethod( iterator_class, "hasPrev", &Falcon::core::Iterator_hasPrev );
   self->addClassMethod( iterator_class, "next", &Falcon::core::Iterator_next );
   self->addClassMethod( iterator_class, "prev", &Falcon::core::Iterator_prev );
   self->addClassMethod( iterator_class, "value", &Falcon::core::Iterator_value ).asSymbol()->
      addParam("subst");
   self->addClassMethod( iterator_class, "key", &Falcon::core::Iterator_key );
   self->addClassMethod( iterator_class, "erase", &Falcon::core::Iterator_erase );
   self->addClassMethod( iterator_class, "compare", &Falcon::core::Iterator_compare ).asSymbol()->
      addParam("item");
   self->addClassMethod( iterator_class, "clone", &Falcon::core::Iterator_clone );
   self->addClassMethod( iterator_class, "find", &Falcon::core::Iterator_find ).asSymbol()->
      addParam("key");
   self->addClassMethod( iterator_class, "insert", &Falcon::core::Iterator_insert ).asSymbol()->
      addParam("key")->addParam("value");

   // ================================================
   // Functional extensions
   //

   //ETA functions
   self->addExtFunc( "all", &Falcon::core::core_all )->setEta( true )->
      addParam("sequence");
   self->addExtFunc( "any", &Falcon::core::core_any )->setEta( true )->
      addParam("sequence");
   self->addExtFunc( "allp", &Falcon::core::core_allp )->setEta( true );
   self->addExtFunc( "anyp", &Falcon::core::core_anyp )->setEta( true );
   self->addExtFunc( "eval", &Falcon::core::core_eval )->setEta( true )->
      addParam("sequence");
   self->addExtFunc( "choice", &Falcon::core::core_choice )->setEta( true )->
      addParam("selector")->addParam("whenTrue")->addParam("whenFalse");
   self->addExtFunc( "xmap", &Falcon::core::core_xmap )->setEta( true )->
      addParam("mfunc")->addParam("sequence");
   self->addExtFunc( "iff", &Falcon::core::core_iff )->setEta( true )->
      addParam("cfr")->addParam("whenTrue")->addParam("whenFalse");
   self->addExtFunc( "lit", &Falcon::core::core_lit )->setEta( true )->
      addParam("item");
   self->addExtFunc( "cascade", &Falcon::core::core_cascade )->setEta( true )->
      addParam("callList");
   self->addExtFunc( "brigade", &Falcon::core::core_brigade )->setEta( true )->
      addParam("fl");
   self->addExtFunc( "dolist", &Falcon::core::core_dolist )->setEta( true )->
      addParam("processor")->addParam("sequence");
   self->addExtFunc( "floop", &Falcon::core::core_floop )->setEta( true )->
      addParam("sequence");
   self->addExtFunc( "firstOf", &Falcon::core::core_firstof )->setEta( true );
   self->addExtFunc( "times", &Falcon::core::core_times )->setEta( true )->
      addParam("count")->addParam("sequence");
   self->addExtFunc( "let", &Falcon::core::core_let )->setEta( true )->
      addParam("dest")->addParam("source");

   // other functions
   self->addExtFunc( "valof", &Falcon::core::core_valof )->addParam("item");
   self->addExtFunc( "min", &Falcon::core::core_min );
   self->addExtFunc( "max", &Falcon::core::core_max );
   self->addExtFunc( "map", &Falcon::core::core_map )->
      addParam("mfunc")->addParam("sequence");
   self->addExtFunc( "filter", &Falcon::core::core_filter )->
      addParam("ffunc")->addParam("sequence");
   self->addExtFunc( "reduce", &Falcon::core::core_reduce )->
      addParam("reductor")->addParam("sequence")->addParam("initial_value");

   self->addExtFunc( "oob", &Falcon::core::core_oob )->
      addParam("item");
   self->addExtFunc( "deoob", &Falcon::core::core_deoob )->
      addParam("item");
   self->addExtFunc( "isoob", &Falcon::core::core_isoob )->
      addParam("item");

   self->addExtFunc( "lbind", &Falcon::core::core_lbind )->
      addParam("name")->addParam("value");


   //=======================================================================
   // RTL basic functionality
   //=======================================================================

   self->addExtFunc( "print", &Falcon::core::print );
   self->addExtFunc( "inspect", &Falcon::core::inspect )->
      addParam("item")->addParam( "depth" )->addParam( "maxLen" )->addParam( "stream" );
   self->addExtFunc( "input", &Falcon::core::input );
   self->addExtFunc( "printl", &Falcon::core::printl );
   self->addExtFunc( "seconds", &Falcon::core::seconds );

   //=======================================================================
   // RTL random api
   //=======================================================================

   self->addExtFunc( "random", &Falcon::core::flc_random );
   self->addExtFunc( "randomChoice", &Falcon::core::flc_randomChoice );
   self->addExtFunc( "randomPick", &Falcon::core::flc_randomPick )->
      addParam("series");
   self->addExtFunc( "randomWalk", &Falcon::core::flc_randomWalk )->
      addParam("series")->addParam("size");
   self->addExtFunc( "randomGrab", &Falcon::core::flc_randomGrab )->
      addParam("series")->addParam("size");
   self->addExtFunc( "randomSeed", &Falcon::core::flc_randomSeed )->
      addParam("seed");
   self->addExtFunc( "randomDice", &Falcon::core::flc_randomDice )->
      addParam("dices");

   //=======================================================================
   // RTL math
   //=======================================================================
   self->addConstant("PI", 3.1415926535897932384626433832795);
   self->addExtFunc( "log", &Falcon::core::flc_math_log )->
      addParam("x");
   self->addExtFunc( "exp", &Falcon::core::flc_math_exp )->
      addParam("x");
   self->addExtFunc( "pow", &Falcon::core::flc_math_pow )->
      addParam("x")->addParam("y");
   self->addExtFunc( "sin", &Falcon::core::flc_math_sin )->
      addParam("x");
   self->addExtFunc( "cos", &Falcon::core::flc_math_cos )->
      addParam("x");
   self->addExtFunc( "tan", &Falcon::core::flc_math_tan )->
      addParam("x");
   self->addExtFunc( "asin", &Falcon::core::flc_math_asin )->
      addParam("x");
   self->addExtFunc( "acos", &Falcon::core::flc_math_acos )->
      addParam("x");
   self->addExtFunc( "atan", &Falcon::core::flc_math_atan )->
      addParam("x");
   self->addExtFunc( "atan2", &Falcon::core::flc_math_atan2 )->
      addParam("x")->addParam("y");
   self->addExtFunc( "rad2deg", &Falcon::core::flc_math_rad2deg )->
      addParam("x");
   self->addExtFunc( "deg2rad", &Falcon::core::flc_math_deg2rad )->
      addParam("x");
   self->addExtFunc( "fract", &Falcon::core::flc_fract )->
      addParam("x");
   self->addExtFunc( "fint", &Falcon::core::flc_fint )->
      addParam("x");
   self->addExtFunc( "round", &Falcon::core::flc_round )->
      addParam("x");
   self->addExtFunc( "floor", &Falcon::core::flc_floor )->
      addParam("x");
   self->addExtFunc( "ceil", &Falcon::core::flc_ceil )->
      addParam("x");
   self->addExtFunc( "abs", Falcon::core::flc_abs )->
      addParam("x");
   self->addExtFunc( "factorial", &Falcon::core::flc_math_factorial )->
      addParam("x");
   self->addExtFunc( "permutations", &Falcon::core::flc_math_permutations )->
      addParam("x")->addParam("y");
   self->addExtFunc( "combinations", &Falcon::core::flc_math_combinations )->
      addParam("x")->addParam("y");

   //=======================================================================
   // RTL complex api
   //=======================================================================
   Falcon::Symbol *c_complex = self->addClass( "Complex", &Falcon::core::Complex_init );
   c_complex->getClassDef()->factory( Falcon::core::Complex_Factory );

   self->addClassProperty( c_complex, "real" );
   self->addClassProperty( c_complex, "imag" );

   self->addClassMethod( c_complex, OVERRIDE_OP_ADD, &Falcon::core::Complex_add__ ).asSymbol()->
      addParam( "complex" );
   self->addClassMethod( c_complex, OVERRIDE_OP_SUB, &Falcon::core::Complex_sub__ ).asSymbol()->
      addParam( "complex" );
   self->addClassMethod( c_complex, OVERRIDE_OP_MUL, &Falcon::core::Complex_mul__ ).asSymbol()->
      addParam( "complex" );
   self->addClassMethod( c_complex, OVERRIDE_OP_DIV, &Falcon::core::Complex_div__ ).asSymbol()->
      addParam( "complex" );
   self->addClassMethod( c_complex, "compare", &Falcon::core::Complex_compare ).asSymbol()->
      addParam( "complex" );
   self->addClassMethod( c_complex, "abs", &Falcon::core::Complex_abs );
   self->addClassMethod( c_complex, "conj", &Falcon::core::Complex_conj );

   self->addClassMethod( c_complex, "toString", &Falcon::core::Complex_toString );

   //=======================================================================
   // RTL string api
   //=======================================================================
   self->addExtFunc( "strFill", &Falcon::core::mth_strFill )
      ->addParam( "string" )->addParam( "chr" );
   self->addExtFunc( "strStartsWith", &Falcon::core::mth_strStartsWith )
      ->addParam( "string" )->addParam( "chr" )->addParam( "icase" );
   self->addExtFunc( "strEndsWith", &Falcon::core::mth_strEndsWith )
      ->addParam( "string" )->addParam( "token" )->addParam( "icase" );
   self->addExtFunc( "strSplit", &Falcon::core::mth_strSplit )
      ->addParam( "string" )->addParam( "token" )->addParam( "count" );
   self->addExtFunc( "strSplitTrimmed", &Falcon::core::mth_strSplitTrimmed )->
      addParam("string")->addParam("token")->addParam("count");
   self->addExtFunc( "strMerge", &Falcon::core::mth_strMerge)->
      addParam("array")->addParam("mergeStr")->addParam("count");
   self->addExtFunc( "strFind", &Falcon::core::mth_strFind )->
      addParam("string")->addParam("needle")->addParam("start")->addParam("end");
   self->addExtFunc( "strBackFind", &Falcon::core::mth_strBackFind )->
      addParam("string")->addParam("needle")->addParam("start")->addParam("end");
   self->addExtFunc( "strReplace", &Falcon::core::mth_strReplace )->
      addParam("string")->addParam("substr")->addParam("repstr")->addParam("start")->addParam("end");
   self->addExtFunc( "strReplicate", &Falcon::core::mth_strReplicate )->
      addParam("string")->addParam("times");

   self->addExtFunc( "strEsq", &Falcon::core::mth_strEsq )->
      addParam("string")->addParam("inplace");
   self->addExtFunc( "strUnesq", &Falcon::core::mth_strUnesq )->
      addParam("string")->addParam("inplace");
   self->addExtFunc( "strEscape", &Falcon::core::mth_strEscape )->
      addParam("string")->addParam("full");
   self->addExtFunc( "strUnescape", &Falcon::core::mth_strUnesq )->
      addParam("string")->addParam("inplace");

   self->addExtFunc( "strBuffer", &Falcon::core::strBuffer )->
      addParam("size");
   self->addExtFunc( "strUpper", &Falcon::core::mth_strUpper )->
      addParam("string");
   self->addExtFunc( "strLower", &Falcon::core::mth_strLower )->
      addParam("string");
   self->addExtFunc( "strCmpIgnoreCase", &Falcon::core::mth_strCmpIgnoreCase )->
      addParam("string1")->addParam("string2");
   self->addExtFunc( "strWildcardMatch", &Falcon::core::mth_strWildcardMatch )
      ->addParam("string")->addParam("wildcard")->addParam("ignoreCase");
   self->addExtFunc( "strToMemBuf", &Falcon::core::mth_strToMemBuf )->
      addParam("string")->addParam("wordWidth");
   self->addExtFunc( "strFromMemBuf", &Falcon::core::strFromMemBuf )->
      addParam("membuf");

   //=======================================================================
   // RTL array API
   //=======================================================================
   self->addExtFunc( "arrayIns", &Falcon::core::mth_arrayIns )->
      addParam("array")->addParam("itempos")->addParam("item");
   self->addExtFunc( "arrayDel", &Falcon::core::mth_arrayDel )->
      addParam("array")->addParam("item");
   self->addExtFunc( "arrayDelAll", &Falcon::core::mth_arrayDelAll )->
      addParam("array")->addParam("item");
   self->addExtFunc( "arrayAdd", &Falcon::core::mth_arrayAdd )->
      addParam("array")->addParam("item");
   self->addExtFunc( "arrayResize", &Falcon::core::mth_arrayResize )->
      addParam("array")->addParam("newSize");
   self->addExtFunc( "arrayScan", &Falcon::core::mth_arrayScan )->
      addParam("array")->addParam("func")->addParam("start")->addParam("end");
   self->addExtFunc( "arrayFind", &Falcon::core::mth_arrayFind )->
      addParam("array")->addParam("item")->addParam("start")->addParam("end");
   self->addExtFunc( "arraySort", &Falcon::core::mth_arraySort )->
      addParam("array")->addParam("sortingFunc");
   self->addExtFunc( "arrayRemove", &Falcon::core::mth_arrayRemove )->
      addParam("array")->addParam("itemPos")->addParam("lastItemPos");
   self->addExtFunc( "arrayMerge", &Falcon::core::mth_arrayMerge )->
      addParam("array1")->addParam("array2")->addParam("insertPos")->addParam("start")->addParam("end");
   self->addExtFunc( "arrayHead", &Falcon::core::mth_arrayHead )->
      addParam("array");
   self->addExtFunc( "arrayTail", &Falcon::core::mth_arrayTail )->
      addParam("array");
   self->addExtFunc( "arrayBuffer", &Falcon::core::arrayBuffer )->
      addParam("size")->addParam("defItem");
   self->addExtFunc( "arrayFill", &Falcon::core::mth_arrayFill )->
      addParam("array")->addParam("item");
   self->addExtFunc( "arrayCompact", &Falcon::core::mth_arrayCompact )->
      addParam("array");
   self->addExtFunc( "arrayNM", &Falcon::core::mth_arrayNM )->
      addParam("array");

   //=======================================================================
   // RTL dictionary
   //=======================================================================
   self->addExtFunc( "bless", &Falcon::core::bless )->
      addParam("dict")->addParam("mode");
   self->addExtFunc( "dictMerge", &Falcon::core::mth_dictMerge )->
      addParam("destDict")->addParam("sourceDict");
   self->addExtFunc( "dictKeys", &Falcon::core::mth_dictKeys )->
      addParam("dict");
   self->addExtFunc( "dictValues", &Falcon::core::mth_dictValues )->
      addParam("dict");
   self->addExtFunc( "dictGet", &Falcon::core::mth_dictGet )->
      addParam("dict")->addParam("key");
   self->addExtFunc( "dictSet", &Falcon::core::mth_dictSet )->
      addParam("dict")->addParam("key")->addParam("value");
   self->addExtFunc( "dictFind", &Falcon::core::mth_dictFind )->
      addParam("dict")->addParam("key");
   self->addExtFunc( "dictBest", &Falcon::core::mth_dictBest )->
      addParam("dict")->addParam("key");
   self->addExtFunc( "dictRemove", &Falcon::core::mth_dictRemove )->
      addParam("dict")->addParam("key");
   self->addExtFunc( "dictClear", &Falcon::core::mth_dictClear )->
      addParam("dict");
   self->addExtFunc( "dictFill", &Falcon::core::mth_dictFill )->
      addParam("dict")->addParam("item");

   self->addExtFunc( "fileType", &Falcon::core::fileType )->
      addParam("filename");
   self->addExtFunc( "fileNameMerge", &Falcon::core::fileNameMerge )->
      addParam("spec")->addParam("path")->addParam("filename")->addParam("ext");
   self->addExtFunc( "fileNameSplit", &Falcon::core::fileNameSplit )->
      addParam("path");
   self->addExtFunc( "fileName", &Falcon::core::fileName )->
      addParam("path");
   self->addExtFunc( "filePath", &Falcon::core::filePath )->
      addParam("fullpath");
   self->addExtFunc( "fileExt", &Falcon::core::fileExt )->
      addParam("fullpath");
   self->addExtFunc( "fileUnit", &Falcon::core::fileUnit )->
      addParam("fullpath");
   self->addExtFunc( "fileMove", &Falcon::core::fileMove )->
      addParam("sourcePath")->addParam("destPath");
   self->addExtFunc( "fileRemove", &Falcon::core::fileRemove )->
      addParam("filename");
   self->addExtFunc( "fileChown", &Falcon::core::fileChown )->
      addParam("path")->addParam("ownerId");
   self->addExtFunc( "fileChmod", &Falcon::core::fileChmod )->
      addParam("path")->addParam("mode");
   self->addExtFunc( "fileChgroup", &Falcon::core::fileChgroup )->
      addParam("path")->addParam("groupId");
   self->addExtFunc( "fileCopy", &Falcon::core::fileCopy )->
      addParam("source")->addParam("dest");

   self->addExtFunc( "dirMake", &Falcon::core::dirMake )->
      addParam("dirname")->addParam("bFull");
   self->addExtFunc( "dirChange", &Falcon::core::dirChange )->
      addParam("newDir");
   self->addExtFunc( "dirCurrent", &Falcon::core::dirCurrent );
   self->addExtFunc( "dirRemove", &Falcon::core::dirRemove )->
      addParam("dir");
   self->addExtFunc( "dirReadLink", &Falcon::core::dirReadLink )->
      addParam("linkPath");
   self->addExtFunc( "dirMakeLink", &Falcon::core::dirMakeLink )->
      addParam("source")->addParam("dest");

   self->addExtFunc( "deserialize", &Falcon::core::deserialize )->
      addParam("stream");

   self->addExtFunc( "include", &Falcon::core::fal_include )->
      addParam("file")->addParam("inputEnc")->addParam("path")->addParam("symDict");

   //==============================================
   // Transcoding functions

   self->addExtFunc( "transcodeTo", &Falcon::core::transcodeTo )->
      addParam("string")->addParam("encoding");
   self->addExtFunc( "transcodeFrom", &Falcon::core::transcodeFrom )->
      addParam("string")->addParam("encoding");
   self->addExtFunc( "getSystemEncoding", &Falcon::core::getSystemEncoding );

   //==============================================
   // Environment variable functions

   self->addExtFunc( "getenv", &Falcon::core::falcon_getenv )->
      addParam("varName");
   self->addExtFunc( "setenv", &Falcon::core::falcon_setenv )->
      addParam("varName")->addParam("value");
   self->addExtFunc( "unsetenv", &Falcon::core::falcon_unsetenv )->
      addParam("varName");
   self->addExtFunc( "getEnviron", &Falcon::core::falcon_getEnviron );

   //=======================================================================
   // Messaging API
   //=======================================================================
   self->addExtFunc( "broadcast", &Falcon::core::broadcast )->
      addParam("msg");
   self->addExtFunc( "subscribe", &Falcon::core::subscribe )->
      addParam("msg")->addParam("handler")->addParam("prio");
   self->addExtFunc( "unsubscribe", &Falcon::core::unsubscribe )->
      addParam("msg")->addParam("handler");
   self->addExtFunc( "getSlot", &Falcon::core::getSlot )->
      addParam("msg")->addParam( "make" );
   self->addExtFunc( "consume", &Falcon::core::consume );
   self->addExtFunc( "assert", &Falcon::core::assert )->
      addParam("msg")->addParam("data");
   self->addExtFunc( "retract", &Falcon::core::retract )->
      addParam("msg");
   self->addExtFunc( "getAssert", &Falcon::core::getAssert )->
      addParam("msg")->addParam( "default" );

   Falcon::Symbol *vmslot_class = self->addClass( "VMSlot", &Falcon::core::VMSlot_init )->
      addParam("name");
   vmslot_class->setWKS( true );
   vmslot_class->getClassDef()->factory( &Falcon::CoreSlotFactory );

   // methods -- the first example is equivalent to the following.
   self->addClassMethod( vmslot_class, "name", &Falcon::core::VMSlot_name );
   self->addClassMethod( vmslot_class, "broadcast", &Falcon::core::VMSlot_broadcast );
   self->addClassMethod( vmslot_class, "subscribe", &Falcon::core::VMSlot_subscribe ).asSymbol()->
         addParam("handler")->addParam("prio");
   self->addClassMethod( vmslot_class, "unsubscribe", &Falcon::core::VMSlot_unsubscribe ).asSymbol()->
         addParam("handler");
   self->addClassMethod( vmslot_class, "prepend", &Falcon::core::VMSlot_prepend ).asSymbol()->
         addParam("handler");
   self->addClassMethod( vmslot_class, "assert", &Falcon::core::VMSlot_assert ).asSymbol()->
         addParam("data");
   self->addClassMethod( vmslot_class, "retract", &Falcon::core::VMSlot_retract );
   self->addClassMethod( vmslot_class, "getAssert", &Falcon::core::VMSlot_getAssert ).asSymbol()->
         addParam("default");
   self->addClassMethod( vmslot_class, "first", &Falcon::core::VMSlot_first );
   self->addClassMethod( vmslot_class, "last", &Falcon::core::VMSlot_last );

   self->addClassMethod( vmslot_class, "send", &Falcon::core::VMSlot_send ).asSymbol()->
         addParam("event");
   self->addClassMethod( vmslot_class, "register", &Falcon::core::VMSlot_register ).asSymbol()->
         addParam("event")->addParam("callback");
   self->addClassMethod( vmslot_class, "getEvent", &Falcon::core::VMSlot_getEvent ).asSymbol()->
         addParam("event");


   //=======================================================================
   // RTL CLASSES
   //=======================================================================

   //==============================================
   // Stream class

   // Factory functions
   self->addExtFunc( "InputStream", &Falcon::core::InputStream_creator )->
      addParam("fileName")->addParam("shareMode");
   self->addExtFunc( "OutputStream", &Falcon::core::OutputStream_creator )->
      addParam("fileName")->addParam("createMode")->addParam("shareMode");
   self->addExtFunc( "IOStream", &Falcon::core::IOStream_creator )->
      addParam("fileName")->addParam("createMode")->addParam("shareMode");

   self->addExtFunc( "readURI", &Falcon::core::readURI )->
      addParam("uri")->addParam("encoding");
   self->addExtFunc( "writeURI", &Falcon::core::writeURI )->
      addParam("uri")->addParam("data")->addParam("encoding");

   // create the stream class (without constructor).
   Falcon::Symbol *stream_class = self->addClass( "Stream" );
   stream_class->setWKS(true);
   //stream_class->getClassDef()->setObjectManager( &core_falcon_data_manager );
   self->addClassMethod( stream_class, "close", &Falcon::core::Stream_close );
   self->addClassMethod( stream_class, "flush", &Falcon::core::Stream_flush );
   self->addClassMethod( stream_class, "read", &Falcon::core::Stream_read ).asSymbol()->
      addParam("buffer")->addParam("size");
   self->addClassMethod( stream_class, "grab", &Falcon::core::Stream_grab ).asSymbol()->
      addParam("size");
   self->addClassMethod( stream_class, "grabLine", &Falcon::core::Stream_grabLine ).asSymbol()->
      addParam("size");
   self->addClassMethod( stream_class, "readLine", &Falcon::core::Stream_readLine ).asSymbol()->
      addParam("buffer")->addParam("size");
   self->addClassMethod( stream_class, "write", &Falcon::core::Stream_write ).asSymbol()->
      addParam("buffer")->addParam("size")->addParam("start");
   self->addClassMethod( stream_class, "seek", &Falcon::core::Stream_seek ).asSymbol()->
      addParam("position");
   self->addClassMethod( stream_class, "seekEnd", &Falcon::core::Stream_seekEnd ).asSymbol()->
      addParam("position");
   self->addClassMethod( stream_class, "seekCur", &Falcon::core::Stream_seekCur ).asSymbol()->
      addParam("position");
   self->addClassMethod( stream_class, "tell", &Falcon::core::Stream_tell );
   self->addClassMethod( stream_class, "truncate", &Falcon::core::Stream_truncate ).asSymbol()->
      addParam("position");
   self->addClassMethod( stream_class, "lastMoved", &Falcon::core::Stream_lastMoved );
   self->addClassMethod( stream_class, "lastError", &Falcon::core::Stream_lastError );
   self->addClassMethod( stream_class, "errorDescription", &Falcon::core::Stream_errorDescription );
   self->addClassMethod( stream_class, "eof", &Falcon::core::Stream_eof );
   self->addClassMethod( stream_class, "isOpen", &Falcon::core::Stream_errorDescription );
   self->addClassMethod( stream_class, "readAvailable", &Falcon::core::Stream_readAvailable ).asSymbol()->
      addParam("seconds");
   self->addClassMethod( stream_class, "writeAvailable", &Falcon::core::Stream_writeAvailable ).asSymbol()->
      addParam("seconds");
   self->addClassMethod( stream_class, "readText", &Falcon::core::Stream_readText ).asSymbol()->
      addParam("buffer")->addParam("size");
   self->addClassMethod( stream_class, "grabText", &Falcon::core::Stream_grabText ).asSymbol()->
      addParam("size");
   self->addClassMethod( stream_class, "writeText", &Falcon::core::Stream_writeText ).asSymbol()->
      addParam("buffer")->addParam("start")->addParam("end");
   self->addClassMethod( stream_class, "setEncoding", &Falcon::core::Stream_setEncoding ).asSymbol()->
      addParam("encoding")->addParam("EOLMode");
   self->addClassMethod( stream_class, "setBuffering", &Falcon::core::Stream_setBuffering ).asSymbol()->
      addParam("size");
   self->addClassMethod( stream_class, "getBuffering", &Falcon::core::Stream_getBuffering );
   self->addClassMethod( stream_class, "clone", &Falcon::core::Stream_clone );

   // Specialization of the stream class to manage the closing of process bound streams.
   Falcon::Symbol *stdstream_class = self->addClass( "StdStream" );
   stdstream_class->setWKS(true);
   self->addClassMethod( stdstream_class, "close", &Falcon::core::StdStream_close );
   self->addClassProperty( stdstream_class, "_stdStreamType" );
   stdstream_class->getClassDef()->addInheritance(  new Falcon::InheritDef( stream_class ) );


   self->addConstant( "FILE_EXCLUSIVE", (Falcon::int64) Falcon::BaseFileStream::e_smExclusive );
   self->addConstant( "FILE_SHARE_READ", (Falcon::int64) Falcon::BaseFileStream::e_smShareRead );
   self->addConstant( "FILE_SHARE", (Falcon::int64) Falcon::BaseFileStream::e_smShareFull );

   self->addConstant( "CR_TO_CR", (Falcon::int64) CR_TO_CR );
   self->addConstant( "CR_TO_CRLF", (Falcon::int64) CR_TO_CRLF );
   self->addConstant( "SYSTEM_DETECT", (Falcon::int64) SYSTEM_DETECT );

   //==============================================
   // StringStream class

   Falcon::Symbol *sstream_ctor = self->addExtFunc( "StringStream._init",
               Falcon::core::StringStream_init, false );
   Falcon::Symbol *sstream_class = self->addClass( "StringStream", sstream_ctor );
   sstream_class->setWKS(true);

   // inherits from stream.
   sstream_class->getClassDef()->addInheritance(  new Falcon::InheritDef( stream_class ) );

   // add methods
   self->addClassMethod( sstream_class, "getString", &Falcon::core::StringStream_getString );
   self->addClassMethod( sstream_class, "closeToString", &Falcon::core::StringStream_closeToString );

   //==============================================
   // The TimeStamp class -- declaration functional equivalent to
   // the one used for StringStream class (there in two steps, here in one).
   Falcon::Symbol *tstamp_class = self->addClass( "TimeStamp", &Falcon::core::TimeStamp_init );
   tstamp_class->setWKS( true );

   // methods -- the first example is equivalent to the following.
   self->addClassMethod( tstamp_class, "currentTime",
      self->addExtFunc( "TimeStamp.currentTime", &Falcon::core::TimeStamp_currentTime, false ) ).setReadOnly(true);

   self->addClassMethod( tstamp_class, "dayOfYear", &Falcon::core::TimeStamp_dayOfYear ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "dayOfWeek", &Falcon::core::TimeStamp_dayOfWeek ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "toString", &Falcon::core::TimeStamp_toString ).setReadOnly(true).asSymbol()->
      addParam("format");
   self->addClassMethod( tstamp_class, "add", &Falcon::core::TimeStamp_add ).setReadOnly(true).asSymbol()->
      addParam("timestamp");
   self->addClassMethod( tstamp_class, "distance", &Falcon::core::TimeStamp_distance ).setReadOnly(true).asSymbol()->
      addParam("timestamp");
   self->addClassMethod( tstamp_class, "isValid", &Falcon::core::TimeStamp_isValid ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "isLeapYear", &Falcon::core::TimeStamp_isLeapYear ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "toLongFormat", &Falcon::core::TimeStamp_toLongFormat ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "fromLongFormat", &Falcon::core::TimeStamp_fromLongFormat ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "compare", &Falcon::core::TimeStamp_compare ).setReadOnly(true).asSymbol()->
      addParam("timestamp");
   self->addClassMethod( tstamp_class, "fromRFC2822", &Falcon::core::TimeStamp_fromRFC2822 ).setReadOnly(true).asSymbol()->
      addParam("sTimestamp");
   self->addClassMethod( tstamp_class, "toRFC2822", &Falcon::core::TimeStamp_toRFC2822 ).setReadOnly(true);
   self->addClassMethod( tstamp_class, "changeZone", &Falcon::core::TimeStamp_changeZone ).setReadOnly(true).asSymbol()->
      addParam("zone");

   // properties
   TimeStamp ts_dummy;
   self->addClassProperty( tstamp_class, "year" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_year );
   self->addClassProperty( tstamp_class, "month" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_month );
   self->addClassProperty( tstamp_class, "day" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_day );
   self->addClassProperty( tstamp_class, "hour" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_hour );
   self->addClassProperty( tstamp_class, "minute" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_minute );
   self->addClassProperty( tstamp_class, "second" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_second );
   self->addClassProperty( tstamp_class, "msec" ).setReflective( e_reflectShort, &ts_dummy, &ts_dummy.m_msec );
   self->addClassProperty( tstamp_class, "timezone" ).
      setReflectFunc( Falcon::core::TimeStamp_timezone_rfrom, &Falcon::core::TimeStamp_timezone_rto );

   Falcon::Symbol *c_timezone = self->addClass( "TimeZone" );
   self->addClassMethod( c_timezone, "getDisplacement", &Falcon::core::TimeZone_getDisplacement ).asSymbol()->
      addParam("tz");
   self->addClassMethod( c_timezone, "describe", &Falcon::core::TimeZone_describe ).asSymbol()->
      addParam("tz");
   self->addClassMethod( c_timezone, "getLocal", &Falcon::core::TimeZone_getLocal );
   self->addClassProperty( c_timezone, "local" ).setInteger( Falcon::tz_local );
   self->addClassProperty( c_timezone, "UTC" ).setInteger( Falcon::tz_UTC );
   self->addClassProperty( c_timezone, "GMT" ).setInteger( Falcon::tz_UTC );
   self->addClassProperty( c_timezone, "E1" ).setInteger( Falcon::tz_UTC_E_1 );
   self->addClassProperty( c_timezone, "E2" ).setInteger( Falcon::tz_UTC_E_2 );
   self->addClassProperty( c_timezone, "E3" ).setInteger( Falcon::tz_UTC_E_3 );
   self->addClassProperty( c_timezone, "E4" ).setInteger( Falcon::tz_UTC_E_4 );
   self->addClassProperty( c_timezone, "E5" ).setInteger( Falcon::tz_UTC_E_5 );
   self->addClassProperty( c_timezone, "E6" ).setInteger( Falcon::tz_UTC_E_6 );
   self->addClassProperty( c_timezone, "E7" ).setInteger( Falcon::tz_UTC_E_7 );
   self->addClassProperty( c_timezone, "E8" ).setInteger( Falcon::tz_UTC_E_8 );
   self->addClassProperty( c_timezone, "E9" ).setInteger( Falcon::tz_UTC_E_9 );
   self->addClassProperty( c_timezone, "E10" ).setInteger( Falcon::tz_UTC_E_10 );
   self->addClassProperty( c_timezone, "E11" ).setInteger( Falcon::tz_UTC_E_11 );
   self->addClassProperty( c_timezone, "E12" ).setInteger( Falcon::tz_UTC_E_12 );

   self->addClassProperty( c_timezone, "W1" ).setInteger( Falcon::tz_UTC_W_1 );
   self->addClassProperty( c_timezone, "W2" ).setInteger( Falcon::tz_UTC_W_2 );
   self->addClassProperty( c_timezone, "W3" ).setInteger( Falcon::tz_UTC_W_3 );
   self->addClassProperty( c_timezone, "W4" ).setInteger( Falcon::tz_UTC_W_4 );
   self->addClassProperty( c_timezone, "EDT" ).setInteger( Falcon::tz_UTC_W_4 );
   self->addClassProperty( c_timezone, "W5" ).setInteger( Falcon::tz_UTC_W_5 );
   self->addClassProperty( c_timezone, "EST" ).setInteger( Falcon::tz_UTC_W_5 );
   self->addClassProperty( c_timezone, "CDT" ).setInteger( Falcon::tz_UTC_W_5 );
   self->addClassProperty( c_timezone, "W6" ).setInteger( Falcon::tz_UTC_W_6 );
   self->addClassProperty( c_timezone, "CST" ).setInteger( Falcon::tz_UTC_W_6 );
   self->addClassProperty( c_timezone, "MDT" ).setInteger( Falcon::tz_UTC_W_6 );
   self->addClassProperty( c_timezone, "W7" ).setInteger( Falcon::tz_UTC_W_7 );
   self->addClassProperty( c_timezone, "MST" ).setInteger( Falcon::tz_UTC_W_7 );
   self->addClassProperty( c_timezone, "PDT" ).setInteger( Falcon::tz_UTC_W_7 );
   self->addClassProperty( c_timezone, "W8" ).setInteger( Falcon::tz_UTC_W_8 );
   self->addClassProperty( c_timezone, "PST" ).setInteger( Falcon::tz_UTC_W_8 );
   self->addClassProperty( c_timezone, "W9" ).setInteger( Falcon::tz_UTC_W_9 );
   self->addClassProperty( c_timezone, "W10" ).setInteger( Falcon::tz_UTC_W_10 );
   self->addClassProperty( c_timezone, "W11" ).setInteger( Falcon::tz_UTC_W_11 );
   self->addClassProperty( c_timezone, "W12" ).setInteger( Falcon::tz_UTC_W_12 );

   self->addClassProperty( c_timezone, "NFT" ).setInteger( Falcon::tz_NFT );
   self->addClassProperty( c_timezone, "ACDT" ).setInteger( Falcon::tz_ACDT );
   self->addClassProperty( c_timezone, "ACST" ).setInteger( Falcon::tz_ACST );
   self->addClassProperty( c_timezone, "HAT" ).setInteger( Falcon::tz_HAT );
   self->addClassProperty( c_timezone, "NST" ).setInteger( Falcon::tz_NST );

   self->addClassProperty( c_timezone, "NONE" ).setInteger( Falcon::tz_NST );

   // A factory function that creates a timestamp already initialized to the current time:
   self->addExtFunc( "CurrentTime", &Falcon::core::CurrentTime );
   self->addExtFunc( "ParseRFC2822", &Falcon::core::ParseRFC2822 );

   //=======================================================================
   // Directory class
   //=======================================================================
   Falcon::Symbol *cont_class = self->addClass( "Continuation", &Falcon::core::Continuation_init )
         ->addParam("item");
   cont_class->getClassDef()->factory( &Falcon::ContinuationCarrier::factory );
   self->addClassMethod( cont_class, "__call", &Falcon::core::Continuation_call );
   self->addClassMethod( cont_class, "reset", &Falcon::core::Continuation_reset );
   self->addClassMethod( cont_class, "complete", &Falcon::core::Continuation_complete );
   self->addClassMethod( cont_class, "_suspend", &Falcon::core::Continuation__suspend );

   //=======================================================================
   // Directory class
   //=======================================================================
   Falcon::Symbol *dir_class = self->addClass( "Directory", &Falcon::core::Directory_init )
         ->addParam("path");
   dir_class->setWKS(true);
   self->addClassMethod( dir_class, "read", &Falcon::core::Directory_read );
   self->addClassMethod( dir_class, "descend", &Falcon::core::Directory_descend ).asSymbol()->
      addParam("dfunc")->addParam("ffunc");
   self->addClassMethod( dir_class, "close", &Falcon::core::Directory_close );
   self->addClassMethod( dir_class, "error", &Falcon::core::Directory_error );

   // Add the directory constants

   //=======================================================================
   // FileStat class
   //=======================================================================

   Falcon::Symbol *fileStats_class = self->addClass( "FileStat",  Falcon::core::FileStat_init )
         ->addParam( "path" );
   fileStats_class->setWKS( true );
   fileStats_class->getClassDef()->factory( &Falcon::core::FileStatObjectFactory );

   // properties
   core::FileStatObject::InnerData id;
   self->addClassProperty( fileStats_class, "ftype" ).
      setReflectFunc( Falcon::core::FileStats_type_rfrom ); // read only, we have no set.
   self->addClassProperty( fileStats_class, "size" ).setReflective( e_reflectLL, &id, &id.m_fsdata.m_size );
   self->addClassProperty( fileStats_class, "owner" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_owner );
   self->addClassProperty( fileStats_class, "group" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_group );
   self->addClassProperty( fileStats_class, "access" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_access );
   self->addClassProperty( fileStats_class, "attribs" ).setReflective( e_reflectUInt, &id, &id.m_fsdata.m_attribs );
   self->addClassProperty( fileStats_class, "mtime" ).
      setReflectFunc( Falcon::core::FileStats_mtime_rfrom );
   self->addClassProperty( fileStats_class, "ctime" ).
      setReflectFunc( Falcon::core::FileStats_ctime_rfrom );
   self->addClassProperty( fileStats_class, "atime" ).
      setReflectFunc( Falcon::core::FileStats_atime_rfrom );

   self->addClassProperty( fileStats_class, "NORMAL" ).setInteger( (Falcon::int64) Falcon::FileStat::t_normal ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "DIR" ).setInteger( (Falcon::int64) Falcon::FileStat::t_dir ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "PIPE" ).setInteger( (Falcon::int64) Falcon::FileStat::t_pipe ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "LINK" ).setInteger( (Falcon::int64) Falcon::FileStat::t_link ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "DEVICE" ).setInteger( (Falcon::int64) Falcon::FileStat::t_device ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "SOCKET" ).setInteger( (Falcon::int64) Falcon::FileStat::t_socket ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "UNKNOWN" ).setInteger( (Falcon::int64) Falcon::FileStat::t_unknown ).
      setReadOnly( true );
   self->addClassProperty( fileStats_class, "NOTFOUND" ).setInteger( (Falcon::int64) Falcon::FileStat::t_notFound ).
      setReadOnly( true );

   // methods - set read only to have full reflection
   self->addClassMethod( fileStats_class, "read",
            Falcon::core::FileStat_read ).setReadOnly(true).asSymbol()->
      addParam("path");

   //=======================================================================
   // The Random class
   //=======================================================================

   Falcon::Symbol *random_class = self->addClass( "Random", &Falcon::core::flc_Random_init )
       ->addParam("seed");
   random_class->setWKS( true );
   self->addClassMethod( random_class, "random", &Falcon::core::flc_random );
   self->addClassMethod( random_class, "randomChoice", &Falcon::core::flc_randomChoice );
   self->addClassMethod( random_class, "randomPick", &Falcon::core::flc_randomPick ).asSymbol()->
       addParam("series");
   self->addClassMethod( random_class, "randomWalk", &Falcon::core::flc_randomWalk ).asSymbol()->
       addParam("series")->addParam("size");
   self->addClassMethod( random_class, "randomGrab", &Falcon::core::flc_randomGrab ).asSymbol()->
       addParam("series")->addParam("size");
   self->addClassMethod( random_class, "randomSeed", &Falcon::core::flc_randomSeed ).asSymbol()->
       addParam("seed");
   self->addClassMethod( random_class, "randomDice", &Falcon::core::flc_randomDice ).asSymbol()->
       addParam("dices");

   //=======================================================================
   // The sequence class
   //=======================================================================
   Falcon::Symbol *sequence_class = self->addClass( "Sequence" );
   sequence_class->exported(false);

   self->addClassMethod( sequence_class, "comp", &Falcon::core::Sequence_comp ).asSymbol()->
      addParam("source")->addParam("filter");
   self->addClassMethod( sequence_class, "mcomp", &Falcon::core::Sequence_mcomp );
   self->addClassMethod( sequence_class, "mfcomp", &Falcon::core::Sequence_mfcomp ).asSymbol()->
      addParam("filter");
   self->addClassMethod( sequence_class, "front", &Falcon::core::Sequence_front );
   self->addClassMethod( sequence_class, "back", &Falcon::core::Sequence_back );
   self->addClassMethod( sequence_class, "last", &Falcon::core::Sequence_last );
   self->addClassMethod( sequence_class, "first", &Falcon::core::Sequence_first );
   self->addClassMethod( sequence_class, "empty", &Falcon::core::Sequence_empty );
   self->addClassMethod( sequence_class, "clear", &Falcon::core::Sequence_clear );
   self->addClassMethod( sequence_class, "append", &Falcon::core::Sequence_append ).asSymbol()->
      addParam("item");
   self->addClassMethod( sequence_class, "prepend", &Falcon::core::Sequence_prepend ).asSymbol()->
      addParam("item");

   //=======================================================================
   // The list class
   //=======================================================================
   Falcon::Symbol *list_class = self->addClass( "List", &Falcon::core::List_init );
   list_class->setWKS(true);
   // inherits from stream.
   list_class->getClassDef()->addInheritance(  new Falcon::InheritDef( sequence_class ) );

   self->addClassMethod( list_class, "push", &Falcon::core::List_push ).asSymbol()->
      addParam("item");
   self->addClassMethod( list_class, "pop", &Falcon::core::List_pop );
   self->addClassMethod( list_class, "pushFront", &Falcon::core::List_pushFront ).asSymbol()->
      addParam("item");
   self->addClassMethod( list_class, "popFront", &Falcon::core::List_popFront );
   self->addClassMethod( list_class, "len", &Falcon::core::List_len );

   //=======================================================================
   // The Set class
   //=======================================================================
   Falcon::Symbol *set_class = self->addClass( "Set", &Falcon::core::Set_init );
   set_class->setWKS(true);
   // inherits from stream.
   set_class->getClassDef()->addInheritance( new Falcon::InheritDef( sequence_class ) );

   self->addClassMethod( set_class, "insert", &Falcon::core::Set_insert ).asSymbol()->
      addParam("item");
   self->addClassMethod( set_class, "remove", &Falcon::core::Set_remove ).asSymbol()->
      addParam("item");
   self->addClassMethod( set_class, "contains", &Falcon::core::Set_contains ).asSymbol()->
      addParam("item");
   self->addClassMethod( set_class, "find", &Falcon::core::Set_find ).asSymbol()->
      addParam("item");
   self->addClassMethod( set_class, "len", &Falcon::core::Set_len );

   //=======================================================================
   // The path class
   //=======================================================================
   Falcon::Symbol *path_class = self->addClass( "Path", &Falcon::core::Path_init )
         ->addParam( "path" );
   path_class->getClassDef()->factory( Falcon::core::PathObjectFactory );
   path_class->setWKS(true);

   self->addClassProperty( path_class, "path" ).
         setReflectFunc( &Falcon::core::Path_path_rfrom, &Falcon::core::Path_path_rto );
   self->addClassProperty( path_class, "unit" ).
         setReflectFunc( &Falcon::core::Path_unit_rfrom, &Falcon::core::Path_unit_rto );
   self->addClassProperty( path_class, "location" ).
         setReflectFunc( &Falcon::core::Path_location_rfrom, &Falcon::core::Path_location_rto );
   self->addClassProperty( path_class, "fulloc" ).
         setReflectFunc( &Falcon::core::Path_fullloc_rfrom, &Falcon::core::Path_fullloc_rto );
   self->addClassProperty( path_class, "file" ).
         setReflectFunc( &Falcon::core::Path_file_rfrom, &Falcon::core::Path_file_rto );
   self->addClassProperty( path_class, "extension" ).
         setReflectFunc( &Falcon::core::Path_extension_rfrom, &Falcon::core::Path_extension_rto );
   self->addClassProperty( path_class, "filename" ).
         setReflectFunc( &Falcon::core::Path_filename_rfrom, &Falcon::core::Path_filename_rto );

   self->addClassProperty( path_class, "winpath" ).
         setReflectFunc( &Falcon::core::Path_winpath_rfrom );
   self->addClassProperty( path_class, "winloc" ).
         setReflectFunc( &Falcon::core::Path_winloc_rfrom );
   self->addClassProperty( path_class, "winfulloc" ).
         setReflectFunc( &Falcon::core::Path_winfulloc_rfrom );

   //=======================================================================
   // The path class
   //=======================================================================
   Falcon::Symbol *uri_class = self->addClass( "URI", &Falcon::core::URI_init )
      ->addParam( "path" )->addParam("decode");
   uri_class->getClassDef()->factory( &Falcon::core::UriObject::factory );
   uri_class->setWKS(true);

   self->addClassProperty( uri_class, "scheme" );
   self->addClassProperty( uri_class, "userInfo" );
   self->addClassProperty( uri_class, "host" );
   self->addClassProperty( uri_class, "port" );
   self->addClassProperty( uri_class, "path" );
   self->addClassProperty( uri_class, "query" );
   self->addClassProperty( uri_class, "fragment" );
   self->addClassProperty( uri_class, "uri" );

   self->addClassMethod( uri_class, "toString", &Falcon::core::URI_toString );
   self->addClassMethod( uri_class, "encode", &Falcon::core::URI_encode ).asSymbol()->
      addParam("string");
   self->addClassMethod( uri_class, "decode", &Falcon::core::URI_decode ).asSymbol()->
      addParam("enc_string");
   self->addClassMethod( uri_class, "getFields", &Falcon::core::URI_getFields );
   self->addClassMethod( uri_class, "setFields", &Falcon::core::URI_setFields ).asSymbol()->
      addParam("fields");

   //=======================================================================
   // The command line parser class
   //=======================================================================

   Falcon::Symbol *cmdparser_class = self->addClass( "CmdlineParser", true );
   self->addClassMethod( cmdparser_class, "parse", &Falcon::core::CmdlineParser_parse ).asSymbol()->
      addParam("args");
   self->addClassMethod( cmdparser_class, "expectValue", &Falcon::core::CmdlineParser_expectValue );
   self->addClassMethod( cmdparser_class, "terminate", &Falcon::core::CmdlineParser_terminate );
   // private property internally used to communicate between the child classes and
   // the base parse.
   self->addClassProperty( cmdparser_class, "_request" );
   // Properties that will hold callbacks
   self->addClassProperty( cmdparser_class, "onOption" );
   self->addClassProperty( cmdparser_class, "onFree" );
   self->addClassProperty( cmdparser_class, "onValue" );
   self->addClassProperty( cmdparser_class, "onSwitchOff" );
   self->addClassProperty( cmdparser_class, "passMinusMinus" );
   self->addClassProperty( cmdparser_class, "lastParsed" );
   self->addClassMethod( cmdparser_class, "usage", &Falcon::core::CmdlineParser_usage );


   //=======================================================================
   // SYSTEM API
   //=======================================================================
   self->addExtFunc( "stdIn", &Falcon::core::_stdIn );
   self->addExtFunc( "stdOut", &Falcon::core::_stdOut );
   self->addExtFunc( "stdErr", &Falcon::core::_stdErr );
   self->addExtFunc( "stdInRaw", &Falcon::core::stdInRaw );
   self->addExtFunc( "stdOutRaw", &Falcon::core::stdOutRaw );
   self->addExtFunc( "stdErrRaw", &Falcon::core::stdErrRaw );
   self->addExtFunc( "systemErrorDescription", &Falcon::core::systemErrorDescription )->
      addParam("errorCode");

   //=======================================================================
   // Tokenizer class
   //=======================================================================
   Falcon::Symbol *tok_class = self->addClass( "Tokenizer",  Falcon::core::Tokenizer_init )
         ->addParam( "seps" )->addParam( "options" )->addParam( "tokLen" )->addParam( "source" );

   // properties
   self->addClassProperty( tok_class, "_source" );
   self->addClassProperty( tok_class, "groupsep" ).setInteger( TOKENIZER_OPT_GRROUPSEP );
   self->addClassProperty( tok_class, "bindsep" ).setInteger( TOKENIZER_OPT_BINDSEP );
   self->addClassProperty( tok_class, "trim" ).setInteger( TOKENIZER_OPT_TRIM );
   self->addClassProperty( tok_class, "retsep" ).setInteger( TOKENIZER_OPT_RSEP );
   self->addClassProperty( tok_class, "wsAsToken" ).setInteger( TOKENIZER_OPT_WSISTOK );

   self->addClassMethod( tok_class, "parse", &Falcon::core::Tokenizer_parse ).asSymbol()->
      addParam("source");
   self->addClassMethod( tok_class, "rewind", &Falcon::core::Tokenizer_rewind );
   self->addClassMethod( tok_class, "next", &Falcon::core::Tokenizer_next );
   self->addClassMethod( tok_class, "nextToken", &Falcon::core::Tokenizer_nextToken );
   self->addClassMethod( tok_class, "token", &Falcon::core::Tokenizer_token );
   self->addClassMethod( tok_class, "hasCurrent", &Falcon::core::Tokenizer_hasCurrent );

   //=======================================================================
   // Table class - tabular programming
   //=======================================================================
   Falcon::Symbol *table_class = self->addClass( "Table", &Falcon::core::Table_init );
   table_class->setWKS(true);

   self->addClassMethod( table_class, "setHeader", &Falcon::core::Table_setHeader ).asSymbol()->
      addParam("header");
   self->addClassMethod( table_class, "getHeader", &Falcon::core::Table_getHeader ).asSymbol()->
      addParam("id");
   self->addClassMethod( table_class, "getColData", &Falcon::core::Table_getColData ).asSymbol()->
      addParam("id");
   self->addClassMethod( table_class, "order", &Falcon::core::Table_order );
   self->addClassMethod( table_class, "len", &Falcon::core::Table_len );
   self->addClassMethod( table_class, "front", &Falcon::core::Table_front );
   self->addClassMethod( table_class, "back", &Falcon::core::Table_back );
   self->addClassMethod( table_class, "first", &Falcon::core::Table_first );
   self->addClassMethod( table_class, "last", &Falcon::core::Table_last );
   self->addClassMethod( table_class, "get", &Falcon::core::Table_get ).asSymbol()->
      addParam("row")->addParam("tcol");
   self->addClassMethod( table_class, OVERRIDE_OP_GETINDEX, &Falcon::core::Table_get ).asSymbol()->
      addParam("row");
   self->addClassMethod( table_class, "set", &Falcon::core::Table_set ).asSymbol()->
      addParam("row")->addParam("element");
   self->addClassMethod( table_class, OVERRIDE_OP_SETINDEX, &Falcon::core::Table_set ).asSymbol()->
      addParam("row")->addParam("element");
   self->addClassMethod( table_class, "columnPos", &Falcon::core::Table_columnPos ).asSymbol()->
      addParam("column");
   self->addClassMethod( table_class, "columnData", &Falcon::core::Table_columnData ).asSymbol()->
      addParam("column")->addParam("data");
   self->addClassMethod( table_class, "find", &Falcon::core::Table_find ).asSymbol()->
      addParam("column")->addParam("value")->addParam("tcol")->addParam("dflt");
   self->addClassMethod( table_class, "insert", &Falcon::core::Table_insert ).asSymbol()->
      addParam("row")->addParam("element");
   self->addClassMethod( table_class, "remove", &Falcon::core::Table_remove ).asSymbol()->
      addParam("row");
   self->addClassMethod( table_class, "append", &Falcon::core::Table_append ).asSymbol()->
      addParam("element");
   self->addClassMethod( table_class, "setColumn", &Falcon::core::Table_setColumn ).asSymbol()->
      addParam("column")->addParam("name")->addParam("coldata");
   self->addClassMethod( table_class, "insertColumn", &Falcon::core::Table_insertColumn ).asSymbol()->
      addParam("column")->addParam("name")->addParam("coldata")->addParam("dflt");
   self->addClassMethod( table_class, "removeColumn", &Falcon::core::Table_removeColumn ).asSymbol()->
      addParam("column");

   self->addClassMethod( table_class, "choice", &Falcon::core::Table_choice ).asSymbol()->
      addParam("func")->addParam("offer")->addParam("rows");
   self->addClassMethod( table_class, "bidding", &Falcon::core::Table_bidding ).asSymbol()->
      addParam("column")->addParam("offer")->addParam("rows");

   self->addClassMethod( table_class, "resetColumn", &Falcon::core::Table_resetColumn ).asSymbol()->
      addParam("column")->addParam("resetVal")->addParam("row")->addParam("value");

   self->addClassMethod( table_class, "pageCount", &Falcon::core::Table_pageCount );
   self->addClassMethod( table_class, "setPage", &Falcon::core::Table_setPage ).asSymbol()->
      addParam("pageId");
   self->addClassMethod( table_class, "curPage", &Falcon::core::Table_curPage );
   self->addClassMethod( table_class, "insertPage", &Falcon::core::Table_insertPage ).asSymbol()->
      addParam("pageId")->addParam("data");
   self->addClassMethod( table_class, "removePage", &Falcon::core::Table_removePage ).asSymbol()->
      addParam("pageId");
   self->addClassMethod( table_class, "getPage", &Falcon::core::Table_getPage ).asSymbol()->
      addParam("pageId");

   return self;
}

}

// Fake an engine-wide module, for external users.

FALCON_MODULE_DECL
{
   return Falcon::core_module_init();
}

/* end of core_module.cpp */

