/*
   FALCON - The Falcon Programming Language.
   FILE: flc_item.h

   Basic Item Api.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 4 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Basic Item Api
*/

#ifndef flc_flc_item_H
#define flc_flc_item_H

#include <stdlib.h>
#include <falcon/types.h>
#include <falcon/itemid.h>
#include <falcon/garbageable.h>
#include <falcon/basealloc.h>
#include <falcon/string.h>
#include <falcon/bommap.h>

namespace Falcon {

class Symbol;
class String;
class CoreObject;
class CoreDict;
class CoreArray;
class CoreClass;
class GarbageItem;
class VMachine;
class Stream;
class Attribute;
class LiveModule;
class MemBuf;

/** Basic item abstraction.*/
class FALCON_DYN_CLASS Item: public BaseAlloc
{
public:
   /** Serialization / deserializzation error code.
      This value is returned by serialization and deserialization
      functions to communicate what whent wrong.
   */
   typedef enum {
      /** All was fine. */
      sc_ok,
      /** File error in serialization/deserialization */
      sc_ferror,
      /** Invalid format in deserialization */
      sc_invformat,
      /** Missing class in deserialization (object cannot be instantiated).*/
      sc_missclass,
      /** Missing symbol in deserialization (requested function not present in VM).*/
      sc_misssym,
      /** VM Error in serialization or de-serialization.
         This is called if VM raised an error during the call of serialization() object
         methods in case serialization is called, or if the VM raises an error during
         init() or deserialization() of deserialized objects.
      */
      sc_vmerror,

      /** Needed VM but missing.
         This serialization or deserialization operation required a VM to be provided,
         but it wasn't.
      */
      sc_missvm
   }
   e_sercode;

private:
   union {
      struct {
         byte methodId;
         byte reserved;
         byte type;
         byte flags;
      } bits;
      uint16 half;
      uint32 whole;
   } m_base;

   union {
      struct {
         int32 val1;
         int32 val2;
      } num ;
      int64 val64;
      numeric number;
      Garbageable *content;
      struct {
         void *voidp;
         void *m_extra;
         LiveModule *m_liveMod;
      };

   } m_data;


   bool internal_is_equal( const Item &other ) const;
   int internal_compare( const Item &other ) const;
   bool serialize_object( Stream *file, const CoreObject *obj, VMachine *vm ) const;
   bool serialize_symbol( Stream *file, const Symbol *sym ) const;
   bool serialize_function( Stream *file, const Symbol *func, VMachine *vm ) const;

   e_sercode deserialize_symbol( Stream *file, VMachine *vm, Symbol **tg_sym, LiveModule **modId );
   e_sercode deserialize_function( Stream *file, VMachine *vm );

#ifdef _MSC_VER
	#if _MSC_VER < 1299
	#define flagOpenRange 0x02
	#define flagIsOob 0x04
	#else
	   static const byte flagOpenRange = 0x02;
	   static const byte flagIsOob = 0x04;
	#endif
#else
	static const byte flagOpenRange = 0x02;
   static const byte flagIsOob = 0x04;
#endif

public:
   Item( byte t=FLC_ITEM_NIL )
   {
      type( t );
   }

   Item( byte t, Garbageable *dt )
   {
      type( t );
      content( dt );
   }

   Item( Symbol *func, LiveModule *mod )
   {
      setFunction( func, mod );
   }

   void setNil() { type( FLC_ITEM_NIL ); }

   Item( const Item &other ) {
      copy( other );
   }

   /** Creates a boolean item. */
   Item( bool tof )
   {
      setBoolean( tof );
   }

   /** Sets this item as boolean */
   void setBoolean( bool tof )
   {
      type( FLC_ITEM_BOOL );
      m_data.num.val1 = tof?1: 0;
   }

   /** Creates an integer item */
   Item( int64 val )
   {
      setInteger( val );
   }


   /** Creates an integer item */
   Item( uint64 val )
   {
      setInteger( (int64)val );
   }

   void setInteger( int64 val ) {
      type(FLC_ITEM_INT);
      m_data.val64 = val;
   }

   /** Creates a numeric item */
   Item( numeric val )
   {
      setNumeric( val );
   }

   void setNumeric( numeric val ) {
      type( FLC_ITEM_NUM );
      m_data.number = val;
   }

   /** Creates a range item */
   Item( int32 val1, int32 val2, bool open )
   {
      setRange( val1, val2, open );
   }

   void setRange( int32 val1, int32 val2, bool open )
   {
      type( FLC_ITEM_RANGE );
      m_data.num.val1 = val1;
      m_data.num.val2 = val2;
      m_base.bits.flags = open? flagOpenRange : 0;
   }


   /** Creates a string item */
   Item( String *str )
   {
      setString( str );
   }

   void setString( String *str ) {
      type( FLC_ITEM_STRING );
      m_data.voidp = str;
   }

   /** Creates an array item */
   Item( CoreArray *array )
   {
      setArray( array );
   }

   void setArray( CoreArray *array ) {
      type( FLC_ITEM_ARRAY );
      m_data.voidp = array;
   }

   /** Creates an object item */
   Item( CoreObject *obj )
   {
      setObject( obj );
   }

   void setObject( CoreObject *obj ) {
      type( FLC_ITEM_OBJECT );
      m_data.voidp = obj;
   }

   /** Creates an attribute. */
   Item( Attribute *attr )
   {
      setAttribute( attr );
   }

   void setAttribute( Attribute *attrib ) {
      type( FLC_ITEM_ATTRIBUTE );
      m_data.voidp = attrib;
   }

   /** Creates a dictionary item */
   Item( CoreDict *obj )
   {
      setDict( obj );
   }

   void setDict( CoreDict *dict ) {
      type( FLC_ITEM_DICT );
      m_data.voidp = dict;
   }

   /** Creates a memory buffer. */
   Item( MemBuf *buf )
   {
      setMemBuf( buf );
   }

   void setMemBuf( MemBuf *b ) {
      type( FLC_ITEM_MEMBUF );
      m_data.voidp = b;
   }

   /** Creates a reference to another item. */
   void setReference( GarbageItem *ref ) {
      type( FLC_ITEM_REFERENCE );
      m_data.voidp = ref;
   }
   GarbageItem *asReference() const { return (GarbageItem *) m_data.voidp; }

   /** Creates a function item */
   void setFunction( Symbol *sym, LiveModule *lmod )
   {
      type( FLC_ITEM_FUNC );
      m_data.voidp = sym;
      m_data.m_liveMod = lmod;
   }

   /** Creates a method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   Item( CoreObject *obj, Symbol *func, LiveModule *lmod )
   {
      setMethod( obj, func, lmod );
   }


   Item( CoreObject *obj, CoreClass *cls )
   {
      setClassMethod( obj, cls );
   }

   /** Creates a method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   void setMethod( CoreObject *obj, Symbol *func, LiveModule *lmod ) {
      type( FLC_ITEM_METHOD );
      m_data.voidp = obj;
      m_data.m_extra = func;
      m_data.m_liveMod = lmod;
   }

   void setClassMethod( CoreObject *obj, CoreClass *cls ) {
      type( FLC_ITEM_CLSMETHOD );
      m_data.voidp = obj;
      m_data.m_extra = cls;
   }

   /** Creates a class item */
   Item( CoreClass *cls )
   {
      setClass( cls );
   }

   void setClass( CoreClass *cls ) {
      type( FLC_ITEM_CLASS );
      // warning: class in extra to be omologue to methodClass()
      m_data.m_extra = cls;
   }

   /** Sets an item as a FBOM.
      Fbom are falcon predefined basic object model methods.
      The original item is copied into this one; the original type is stored
      in the "reserved" field, while the method is stored as an integer
      between 0 and 255 in methodId.
   */

   void setFbom( const Item &original, byte methodId )
   {
      m_data = original.m_data;
      m_base.bits.flags = original.m_base.bits.flags;
      m_base.bits.methodId = methodId;
      m_base.bits.reserved = original.m_base.bits.type;
      m_base.bits.type = FLC_ITEM_FBOM;
   }

   /** Defines this item as a out of band data.
      Out of band data allow out-of-order sequencing in functional programming.
      If an item is out of band, it's type it's still the original one, and
      its value is preserved across function calls and returns; however, the
      out of band data may innescate a meta-level processing of the data
      travelling through the functions.

      In example, returning an out of band NIL value from an xmap mapping
      function will cause xmap to discard the data.
   */
   void setOob() { m_base.bits.flags |= flagIsOob; }

   /** Clear out of band status of this item.
      \see setOob()
   */
   void resetOob() { m_base.bits.flags &= ~flagIsOob; }

   /** Sets or clears the out of band status status of this item.
      \param oob true to make this item out of band.
      \see setOob()
   */
   void setOob( bool oob ) {
      if ( oob )
         m_base.bits.flags |= flagIsOob;
      else
         m_base.bits.flags &= ~flagIsOob;
   }
   /** Set this item as a lightweight pointer.

      Lightweight pointers are used in "perfect reflection".
      They point to a certain data memory, which exists beyond
      their scope, and once queried they can create adequate
      items either directly or through a builder function.

      This version of the function sets the target type to be of a certain size,
      the target data is turned into a
   */
   void setLightPointer( void *memory, uint8 size )
   {
   }

   /** Set this item as a user-defined pointers.
      Used for some two-step extension functions.
      They are completely user managed, and the VM never provides any
      help to handle them.
   */
   void setUserPointer( void *tpd )
   {
      type( FLC_ITEM_POINTER );
      m_data.voidp = tpd;
   }

   void *asUserPointer()
   {
      return m_data.voidp;
   }

   bool isUserPointer() const { return m_base.bits.flags == FLC_ITEM_POINTER; }

   /** Tells wether this item is out of band.
      \return true if out of band.
      \see oob()
   */
   bool isOob() const { return (m_base.bits.flags & flagIsOob )== flagIsOob; }

   byte type() const { return m_base.bits.type; }
   void type( byte nt ) { m_base.bits.flags = 0; m_base.bits.type = nt; }

   /** Returns the content of the item */
   Garbageable *content() const { return m_data.content; }

   void content( Garbageable *dt ) {
      m_data.content = dt;
   }

   void copy( const Item &other )
   {
      m_base = other.m_base;
      m_data = other.m_data;
   }

   /** Tells if this item is callable.
      This function will turn this object into a nil
      if the item referenced a dead module. As this is a pathological
      situation, a const cast is forced.
   */
   bool isCallable() const;

   bool isOrdinal() const {
      return type() == FLC_ITEM_INT || type() == FLC_ITEM_NUM;
   }

   bool asBoolean() const
   {
      return m_data.num.val1 != 0;
   }

   int64 asInteger() const { return m_data.val64; }

   numeric asNumeric() const { return m_data.number; }

   int32 asRangeStart() const { return m_data.num.val1; }
   int32 asRangeEnd()  const { return m_data.num.val2; }
   bool asRangeIsOpen()  const { return (m_base.bits.flags & flagOpenRange) != 0; }

   String *asString() const { return (String *) m_data.voidp; }
   /** Provides a basic string representation of the item.
      Use Falcon::Format for a finer control of item representation.
      \param target a string where the item string representation will be placed.
   */
   void toString( String &target ) const;
   CoreArray *asArray() const { return (CoreArray *) m_data.voidp; }
   CoreObject *asObject() const { return (CoreObject *) m_data.voidp; }
   CoreDict *asDict() const { return ( CoreDict *) m_data.voidp; }
   MemBuf *asMemBuf() const { return ( MemBuf *) m_data.voidp; }

   CoreClass *asClass() const { return (CoreClass *) m_data.m_extra; }
   Symbol *asFunction() const { return (Symbol *) m_data.voidp; }

   CoreObject *asMethodObject() const { return (CoreObject *) m_data.voidp; }
   Symbol *asMethodFunction() const { return (Symbol *)m_data.m_extra; }
   CoreClass *asMethodClass() const { return (CoreClass*) m_data.m_extra; }
   Attribute *asAttribute() const { return (Attribute *) m_data.voidp; }

   LiveModule *asModule() const { return m_data.m_liveMod; }

   /** Convert current object into an integer.
      This operations is usually done on integers, numeric and strings.
      It will do nothing meaningfull on other types.
   */
   int64 forceInteger() const ;

   /** Convert current object into a numeric.
      This operations is usually done on integers, numeric and strings.
      It will do nothing meaningfull on other types.
   */
   numeric forceNumeric() const ;

   /** Calculates the hash function for this item.
   */
   uint32 hash() const;

   bool isNil() const { return type() == FLC_ITEM_NIL; }
   bool isBoolean() const { return type() == FLC_ITEM_BOOL; }
   bool isInteger() const { return type() == FLC_ITEM_INT; }
   bool isNumeric() const { return type() == FLC_ITEM_NUM; }
   bool isScalar() const { return type() == FLC_ITEM_INT || type() == FLC_ITEM_NUM; }
   bool isRange() const { return type() == FLC_ITEM_RANGE; }
   bool isString() const { return type() == FLC_ITEM_STRING; }
   bool isArray() const { return type() == FLC_ITEM_ARRAY; }
   bool isDict() const { return type() == FLC_ITEM_DICT; }
   bool isMemBuf() const { return type() == FLC_ITEM_MEMBUF; }
   bool isObject() const { return type() == FLC_ITEM_OBJECT; }
   bool isReference() const { return type() == FLC_ITEM_REFERENCE; }
   bool isFunction() const { return type() == FLC_ITEM_FUNC; }
   bool isMethod() const { return type() == FLC_ITEM_METHOD; }
   bool isClassMethod() const { return type() == FLC_ITEM_CLSMETHOD; }
   bool isClass() const { return type() == FLC_ITEM_CLASS; }
   bool isFbom() const { return type() == FLC_ITEM_FBOM; }
   bool isAttribute() const { return type() == FLC_ITEM_ATTRIBUTE; }

   void getFbomItem( Item &target ) const
   {
      target.m_data = m_data;
      target.m_base.bits.type = m_base.bits.reserved;
      target.m_base.bits.flags = m_base.bits.flags;
   }

   byte getFbomMethod() const { return m_base.bits.methodId; }


   bool equal( const Item &other ) const {
      if ( type() == other.type() )
         return internal_is_equal( other );
      else if ( type() == FLC_ITEM_INT && other.type() == FLC_ITEM_NUM ) {
         return ((numeric) m_data.num.val1) == other.m_data.number;
      }
      else if ( type() == FLC_ITEM_NUM && other.type() == FLC_ITEM_INT ) {
         return ((numeric) other.m_data.num.val1) == m_data.number;
      }
      return false;
   }

   int compare( const Item &other ) const;

   bool isOfClass( const String &className ) const;

   bool isTrue() const;

   Item &operator=( const Item &other ) { copy( other ); return *this; }
   bool operator==( const Item &other ) const { return equal(other); }
   bool operator!=( const Item &other ) const { return !equal(other); }
   bool operator<(const Item &other) const { return compare( other ) < 0; }
   bool operator<=(const Item &other) const { return compare( other ) <= 0; }
   bool operator>(const Item &other) const { return compare( other ) > 0; }
   bool operator>=(const Item &other) const { return compare( other ) >= 0; }

   Item *dereference();
   const Item *dereference() const;

   /** To be used by function that are taking items outside the VM */
   void destroy();

   /** Turns this item in a method of the given object.
      This is meant to be used by external functions when accessing object properties.
      VM always creates a method when accessor is used; in example, myObject.someFunc()
      will create a method myObject.someFunc if there is some callable inside the given
      property, and then will call it. In this way, a function may be assigned to that
      property, and the VM will take care to create an item that will turn the function
      in a method.

      But when a non-vm program accesses an object "method", the calling program may
      have assigned something different to it in the meanwhile, and what its returned
      in CoreObject::getProperty() won't be a callable method, but just the assigned
      object.

      Methodize will turn such an item in a callable method of the object given as
      parameter, if this is possible, else it will return false.
      \note External methods (that is, methods calling external functions) won't be
      methodized, as they often rely in external values carried inside their CoreObject
      owners. However, the function will return true, as the object can be called, and
      very probably it will do what expected by the user (as the external method would
      have never used anything other than the VM generated self anyhow).
      \note CoreObject::getMethod() is a shortcut to this function.

      \param self the object that will be set as "self" for the method
      \return true if the item can be called properly, false if it's not a callable.
   */
   bool methodize( const CoreObject *self );


   /** Serialize this item.
      This method stores an item on a stream for later retrival. Providing a virtual
      machine that is optional; if not provided, items requiring the VM for serialization
      won't be correctly serialized. In example, objects may provide a serialize() method;
      if that method exists, the VM would call it in order to provide the object with its own
      personalized serialization.

      Unless certain that the given object is not an object with a serialized method,
      or does not references one at any level, this function should always be called with
      a VM ready.

      The function can return true if the serialization succeded. It will return false if
      the serialization failed; in that case, if there is an error on the stream, it can
      be retreived form the stream variable. If the stream reports no error, then the
      serialization failed because a VM was not provided while a serialized method should
      have been called on the target object, or in any referenced object, or because the
      VM received an error during the method call.

      \param out the output stream
      \param vm the virtual machine that can be used for object serialization
      \return an error code in case of error (\see e_sercode).
   */
   e_sercode serialize( Stream *out, VMachine *vm = 0 ) const;


   /** Loads a serialized item from a stream.
      This method restores an item previously stored on a stream for later retrival.
      Providing a virtual
      machine that is optional; if not provided, items requiring the VM for deserialization
      won't be correctly restored. Objects deserialization requires a VM readied with
      the class the object derives from.
      \see serialize()

      \param in the input stream
      \param vm the virtual machine that can be used for object deserialization
      \return an error code in case of error (\see e_sercode).
   */
   e_sercode deserialize( Stream *in, VMachine *vm = 0 );

   /** Flags, used for internal vm needs. */
   byte flags() const { return m_base.bits.flags; }
   void flags( byte b ) { m_base.bits.flags = b; }
   void flagsOn( byte b ) { m_base.bits.flags |= b; }
   void flagsOff( byte b ) { m_base.bits.flags &= ~b; }


   /** Clone the item (with the help of a VM).
      If the item is not cloneable, the method returns false. Is up to the caller to
      raise an appropriate error if that's the case.
      The VM parameter may be zero; in that case, returned items will not be stored in
      any garbage collector.

      Reference items are de-referenced; if cloning a reference, the caller will obtain
      a clone of the target item, not a clone of the reference.

      Also, in that case, the returned item will be free of reference.

      \param vm the virtual machine used for cloning.
      \param target the item where to stored the cloned instance of this item.
      \return true if the clone operation is possible
   */
   bool clone( Item &target, VMachine *vm ) const;

   /** Returns a Falcon Basic Object Model method for the given object.
      If the given item provides the named FBOM method, the function returns true
      and the item "method" is set to a correctly setup FBOM item.

      \param property the name of the searched property
      \param method on success, a valorized FBOM item
      \return true if the property is a FBOM property name
   */
   bool getBom( const String &property, Item &method, BomMap *bmap ) const;

   /** Call this item's basic object method, if the item is a FBOM

      \param vm the VM that will be used to call this bom.
   */
   bool callBom( VMachine *vm ) const;

   /** Return true if the item deep.
      Deep items are the ones that are subject to garbage collecting.
      \return true if the item is deep.
   */
   bool isDeep() const { return type() < FLC_ITEM_FIRST_DEEP; }
};

/** Creates a garbageable version of an item.
   This class repeats the structure of an item holding an instance
   of it, but it derives from Garbageable. This makes it a vessel
   for item references.

   It must be created by a MemPool with the MemPool::referenceItem()
   method.
*/
class FALCON_DYN_CLASS GarbageItem: public Garbageable
{
   Item m_item;

public:
   GarbageItem( VMachine *vm, const Item &origin ):
      Garbageable( vm, sizeof( this ) ),
      m_item( origin )
   {}

   virtual ~GarbageItem() {};

   /** Returns the item part stored in this garbage item.
      \return the held item.
   */
   const Item &origin() const { return m_item; }

   /** Returns the item part stored in this garbage item.
      \return the held item.
   */
   Item &origin() { return m_item; }
};

inline Item *Item::dereference()
{
   if ( type() != FLC_ITEM_REFERENCE )
      return this;
   return &asReference()->origin();
};

inline const Item *Item::dereference() const
{
   if ( type() != FLC_ITEM_REFERENCE )
      return this;
   return &asReference()->origin();
};

}

#endif
/* end of flc_item.h */
