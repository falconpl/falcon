/*
   FALCON - The Falcon Programming Language.
   FILE: flc_item.h

   Basic Item Api.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ott 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
class GarbagePointer;
class FalconData;


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
      } ptr;

      struct {
         int32 rstart;
         int32 rend;
         int32 rstep;
      } rng;

      struct {
         GarbagePointer *gcptr;
         int32 signature;
      } gptr;
   } m_data;

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



   bool internal_is_equal( const Item &other ) const;
   int internal_compare( const Item &other ) const;
   bool serialize_object( Stream *file, CoreObject *obj, bool bLive ) const;
   bool serialize_symbol( Stream *file, const Symbol *sym ) const;
   bool serialize_function( Stream *file, const Symbol *func ) const;
   bool serialize_class( Stream *file, const CoreClass *cls ) const;

   e_sercode deserialize_symbol( Stream *file, VMachine *vm, Symbol **tg_sym, LiveModule **modId );
   e_sercode deserialize_function( Stream *file, VMachine *vm );
   e_sercode deserialize_class( Stream *file, VMachine *vm );

#ifdef _MSC_VER
	#if _MSC_VER < 1299
	#define flagOpenRange 0x02
	#define flagIsOob 0x04
	#define flagFuture 0x08
	#else
	   static const byte flagOpenRange = 0x02;
	   static const byte flagIsOob = 0x04;
	   static const byte flagFuture = 0x08;
	#endif
#else
	static const byte flagOpenRange = 0x02;
   static const byte flagIsOob = 0x04;
   static const byte flagFuture = 0x08;
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

   Item( const Symbol *func, LiveModule *mod )
   {
      setFunction( func, mod );
   }

   void setNil() { type( FLC_ITEM_NIL ); }

   Item( const Item &other ) {
      copy( other );
   }

   /** Creates a boolean item. */

   /** Sets this item as boolean */
   void setBoolean( bool tof )
   {
      type( FLC_ITEM_BOOL );
      m_data.num.val1 = tof?1: 0;
   }

   /** Creates an integer item */
   Item( int16 val )
   {
      setInteger( (int64) val );
   }

   /** Creates an integer item */
   Item( uint16 val )
   {
      setInteger( (int64) val );
   }

   /** Creates an integer item */
   Item( int32 val )
   {
      setInteger( (int64) val );
   }

   /** Creates an integer item */
   Item( uint32 val )
   {
      setInteger( (int64) val );
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
      m_data.rng.rstart = val1;
      m_data.rng.rend = val2;
      m_data.rng.rstep = 0;
      m_base.bits.flags = open? flagOpenRange : 0;
   }

   void setRange( int32 val1, int32 val2, int32 step, bool open )
   {
      type( FLC_ITEM_RANGE );
      m_data.rng.rstart = val1;
      m_data.rng.rend = val2;
      m_data.rng.rstep = step;
      m_base.bits.flags = open? flagOpenRange : 0;
   }


   /** Creates a string item */
   Item( String *str )
   {
      setString( str );
   }

   void setString( String *str ) {
      type( FLC_ITEM_STRING );
      m_data.ptr.voidp = str;
   }

   /** Creates an array item */
   Item( CoreArray *array )
   {
      setArray( array );
   }

   void setArray( CoreArray *array ) {
      type( FLC_ITEM_ARRAY );
      m_data.ptr.voidp = array;
   }

   /** Creates an object item */
   Item( CoreObject *obj )
   {
      setObject( obj );
   }

   void setObject( CoreObject *obj ) {
      type( FLC_ITEM_OBJECT );
      m_data.ptr.voidp = obj;
   }

   /** Creates an attribute. */
   Item( Attribute *attr )
   {
      setAttribute( attr );
   }

   void setAttribute( Attribute *attrib ) {
      type( FLC_ITEM_ATTRIBUTE );
      m_data.ptr.voidp = attrib;
   }

   /** Creates a dictionary item */
   Item( CoreDict *obj )
   {
      setDict( obj );
   }

   void setDict( CoreDict *dict ) {
      type( FLC_ITEM_DICT );
      m_data.ptr.voidp = dict;
   }

   /** Creates a memory buffer. */
   Item( MemBuf *buf )
   {
      setMemBuf( buf );
   }

   void setMemBuf( MemBuf *b ) {
      type( FLC_ITEM_MEMBUF );
      m_data.ptr.voidp = b;
   }

   /** Creates a reference to another item. */
   void setReference( GarbageItem *ref ) {
      type( FLC_ITEM_REFERENCE );
      m_data.ptr.voidp = ref;
   }
   GarbageItem *asReference() const { return (GarbageItem *) m_data.ptr.voidp; }

   /** Creates a function item */
   void setFunction( const Symbol *sym, LiveModule *lmod )
   {
      type( FLC_ITEM_FUNC );
      m_data.ptr.voidp = const_cast<Symbol *>(sym);
      m_data.ptr.m_liveMod = lmod;
   }

   /** Creates a late binding item.
      The late binding is just a string in a live module which is
      resolved into a value by referencing a item in the current
      context (symbol tables) having the given name at runtime.

      Thus, the string representing the late binding symbol name
      lives in the live module that generated this LBind. If
      the module is unloaded, the LBind is invalidated.

      \param lbind The name of the late binding symbol.
      \param val If provided, a future value (future binding).
   */
   void setLBind( String *lbind, GarbageItem *val=0 )
   {
      type( FLC_ITEM_LBIND );
      m_data.ptr.voidp = lbind;
      m_data.ptr.m_extra = val;
   }

   /** Returns true if this item is a valid LBind.
   */
   bool isLBind() const { return type() == FLC_ITEM_LBIND; }
   bool isFutureBind() const { return isLBind() && m_data.ptr.m_extra != 0; }

   /** Return the binding name associate with this LBind item.
   */
   String *asLBind() const { return (String *) m_data.ptr.voidp; }
   GarbageItem *asFBind() const { return (GarbageItem *) m_data.ptr.m_extra; }

   const Item &asFutureBind() const;
   Item &asFutureBind();

   /** Creates a method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   Item( CoreObject *obj, const Symbol *func, LiveModule *lmod )
   {
      setMethod( obj, func, lmod );
   }

   /** Creates a table/array method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   Item( CoreArray *arr, const Symbol *func, LiveModule *lmod )
   {
      setTabMethod( arr, func, lmod );
   }

   Item( CoreDict *dict, const Symbol *func, LiveModule *lmod )
   {
      setTabMethod( dict, func, lmod );
   }


   Item( CoreObject *obj, CoreClass *cls )
   {
      setClassMethod( obj, cls );
   }

   /** Creates a method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   void setMethod( CoreObject *obj, const Symbol *func, LiveModule *lmod ) {
      type( FLC_ITEM_METHOD );
      m_data.ptr.voidp = obj;
      m_data.ptr.m_extra = const_cast<Symbol *>(func);
      m_data.ptr.m_liveMod = lmod;
   }

   /** Creates a table/array method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   void setTabMethod( CoreArray *arr, const Symbol *func, LiveModule *lmod ) {
      type( FLC_ITEM_TABMETHOD );
      m_base.bits.reserved = 0;
      m_data.ptr.voidp = arr;
      m_data.ptr.m_extra = const_cast<Symbol *>(func);
      m_data.ptr.m_liveMod = lmod;
   }

   void setTabMethod( CoreDict *dict, const Symbol *func, LiveModule *lmod ) {
      type( FLC_ITEM_TABMETHOD );
      m_base.bits.reserved = 1;
      m_data.ptr.voidp = dict;
      m_data.ptr.m_extra = const_cast<Symbol *>(func);
      m_data.ptr.m_liveMod = lmod;
   }


   void setClassMethod( CoreObject *obj, CoreClass *cls ) {
      type( FLC_ITEM_CLSMETHOD );
      m_data.ptr.voidp = obj;
      m_data.ptr.m_extra = cls;
   }

   /** Creates a class item */
   Item( CoreClass *cls )
   {
      setClass( cls );
   }

   void setClass( CoreClass *cls ) {
      type( FLC_ITEM_CLASS );
      // warning: class in extra to be omologue to methodClass()
      m_data.ptr.m_extra = cls;
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

   /** Set this item as a user-defined pointers.
      Used for some two-step extension functions.
      They are completely user managed, and the VM never provides any
      help to handle them.
   */
   void setUserPointer( void *tpd )
   {
      type( FLC_ITEM_USER_POINTER );
      m_data.ptr.voidp = tpd;
   }

   void *asUserPointer() const
   {
      return m_data.ptr.voidp;
   }

   bool isUserPointer() const { return type() == FLC_ITEM_USER_POINTER; }

   /** Set this item as a user-defined Garbage pointers.
       VM provides GC-control over them.
   */
   void setGCPointer( VMachine *vm, FalconData *ptr, uint32 sig = 0 );
   void setGCPointer( GarbagePointer *shell, uint32 sig = 0 );

   FalconData *asGCPointer() const;
   GarbagePointer *asGCPointerShell() const { return m_data.gptr.gcptr; }
   uint32 asGCPointerSignature()  const { return m_data.gptr.signature; }

   bool isGCPointer() const { return type() == FLC_ITEM_GCPTR; }

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
      #if defined( __sparc )
        //memcpy( this, &other, sizeof(Item) );
        m_base.whole = other.m_base.whole;
	m_data.ptr.voidp = other.m_data.ptr.voidp;
        m_data.ptr.m_extra = other.m_data.ptr.m_extra;
        m_data.ptr.m_liveMod = other.m_data.ptr.m_liveMod;
      #else
        m_base = other.m_base;
        m_data = other.m_data;
      #endif
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

   int32 asRangeStart() const { return m_data.rng.rstart; }
   int32 asRangeEnd()  const { return m_data.rng.rend; }
   int32 asRangeStep()  const { return m_data.rng.rstep; }
   bool asRangeIsOpen()  const { return (m_base.bits.flags & flagOpenRange) != 0; }

   String *asString() const { return (String *) m_data.ptr.voidp; }
   /** Provides a basic string representation of the item.
      Use Falcon::Format for a finer control of item representation.
      \param target a string where the item string representation will be placed.
   */
   void toString( String &target ) const;
   CoreArray *asArray() const { return (CoreArray *) m_data.ptr.voidp; }
   CoreObject *asObject() const { return (CoreObject *) m_data.ptr.voidp; }
   CoreDict *asDict() const { return ( CoreDict *) m_data.ptr.voidp; }
   MemBuf *asMemBuf() const { return ( MemBuf *) m_data.ptr.voidp; }

   CoreClass *asClass() const { return (CoreClass *) m_data.ptr.m_extra; }
   const Symbol *asFunction() const { return (const Symbol *) m_data.ptr.voidp; }

   CoreObject *asMethodObject() const { return (CoreObject *) m_data.ptr.voidp; }
   CoreArray *asTabMethodArray() const { return (CoreArray *) m_data.ptr.voidp; }
   CoreDict *asTabMethodDict() const { return (CoreDict *) m_data.ptr.voidp; }
   bool isTabMethodDict() const { return m_base.bits.reserved==1; }
   const Symbol *asMethodFunction() const { return (const Symbol *)m_data.ptr.m_extra; }
   CoreClass *asMethodClass() const { return (CoreClass*) m_data.ptr.m_extra; }
   Attribute *asAttribute() const { return (Attribute *) m_data.ptr.voidp; }

   LiveModule *asModule() const { return m_data.ptr.m_liveMod; }

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
   bool isTabMethod() const { return type() == FLC_ITEM_TABMETHOD; }
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
      This method stores an item on a stream for later retrival.

      The function can return true if the serialization succeded. It will return false if
      the serialization failed; in that case, if there is an error on the stream, it can
      be retreived form the stream variable. If the stream reports no error, then the
      serialization failed because a VM was not provided while a serialized method should
      have been called on the target object, or in any referenced object, or because the
      VM received an error during the method call.

      \param out the output stream
      \param bLive true
      \return an error code in case of error (\see e_sercode).
   */
   e_sercode serialize( Stream *out, bool bLive = false ) const;


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
   bool isDeep() const { return type() >= FLC_ITEM_FIRST_DEEP; }
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
