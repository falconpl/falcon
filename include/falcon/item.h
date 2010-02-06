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
#include <falcon/corerange.h>

#define OVERRIDE_OP_ADD       "__add"
#define OVERRIDE_OP_SUB       "__sub"
#define OVERRIDE_OP_MUL       "__mul"
#define OVERRIDE_OP_DIV       "__div"
#define OVERRIDE_OP_MOD       "__mod"
#define OVERRIDE_OP_POW       "__pow"
#define OVERRIDE_OP_NEG       "__neg"
#define OVERRIDE_OP_INC       "__inc"
#define OVERRIDE_OP_DEC       "__dec"
#define OVERRIDE_OP_INCPOST   "__incpost"
#define OVERRIDE_OP_DECPOST   "__decpost"
#define OVERRIDE_OP_CALL      "__call"
#define OVERRIDE_OP_GETINDEX  "__getIndex"
#define OVERRIDE_OP_SETINDEX  "__setIndex"

namespace Falcon {

class Symbol;
class CoreString;
class CoreObject;
class CoreDict;
class CoreArray;
class CoreClass;
class CallPoint;
class CoreFunc;
class GarbageItem;
class VMachine;
class Stream;
class LiveModule;
class MemBuf;
class GarbagePointer;
class FalconData;
class CodeError;
class DeepItem;


typedef void** CommOpsTable;
extern FALCON_DYN_SYM CommOpsTable CommOpsDict[];


/** Basic item abstraction.*/
class FALCON_DYN_CLASS Item: public BaseAlloc
{
public:

   /** Common operations that can be performed on items.
      Each item type has a function pointer table taking care of this
      operations.

      Deep items operations is that of searching for overloadings
      via the deep item common DeepItem::getProperty() method,
      and then eventually calling the operator implementation.

      The operator implementation is called on the VM instance of the
      deep item.
   */
   typedef enum {
      co_add,
      co_sub,
      co_mul,
      co_div,
      co_mod,
      co_pow,

      co_neg,

      co_inc,
      co_dec,
      co_incpost,
      co_decpost,

      co_compare,

      co_getIndex,
      co_setIndex,
      co_getProperty,
      co_setProperty,

      co_call

   } e_commops;

   /** Serialization / deserialization error code.
      This value is returned by serialization and deserialization
      functions to communicate what went wrong.
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
      sc_missvm,
      /** Hit EOF while de-serializing */
      sc_eof
   }
   e_sercode;

protected:
  union {
      struct {
         union {
            int32 val32;
            int64 val64;
            numeric number;
            Garbageable *content;

            struct {
               void *voidp;
               void *extra;
            } ptr;

         } data;

         CallPoint *method;

         union {
            struct {
               byte type;
               byte flags;
               byte oldType;
               byte reserved;
            } bits;
            uint16 half;
            uint32 whole;
         } base;
      } ctx;

      struct {
         uint64 low;
         uint64 high;
      } parts;
   } all;


   bool serialize_object( Stream *file, CoreObject *obj, bool bLive ) const;
   bool serialize_symbol( Stream *file, const Symbol *sym ) const;
   bool serialize_function( Stream *file, const CoreFunc *func, bool bLive ) const;
   bool serialize_class( Stream *file, const CoreClass *cls ) const;

   e_sercode deserialize_symbol( Stream *file, VMachine *vm, Symbol **tg_sym, LiveModule **modId );
   e_sercode deserialize_function( Stream *file, VMachine *vm );
   e_sercode deserialize_class( Stream *file, VMachine *vm );

#ifdef _MSC_VER
	#if _MSC_VER < 1299
	#define flagIsMethodic 0x02
	#define flagIsOob 0x04
	#define flagLiteral 0x08
	#else
	   static const byte flagIsMethodic = 0x02;
	   static const byte flagIsOob = 0x04;
	   static const byte flagLiteral = 0x08;
	#endif
#else
   static const byte flagIsMethodic = 0x02;
   static const byte flagIsOob = 0x04;
   static const byte flagLiteral = 0x08;
#endif

public:
   Item( byte t=FLC_ITEM_NIL )
   {
      type( t );
   }

   Item( byte t, Garbageable *dt );


   Item( CoreFunc* cf )
   {
      setFunction( cf );
   }

   void setNil() { type( FLC_ITEM_NIL ); }

   void setUnbound() { type( FLC_ITEM_UNB ); }

   Item( const Item &other ) {
      copy( other );
   }

   /** Creates a boolean item. */

   /** Sets this item as boolean */
   void setBoolean( bool tof )
   {
      type( FLC_ITEM_BOOL );
      all.ctx.data.val32 = tof? 1: 0;
   }

   /** Creates an garbage pointer item */
   Item( GarbagePointer* val )
   {
      setGCPointer( val );
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
      all.ctx.data.val64 = val;
   }

   /** Creates a numeric item */
   Item( numeric val )
   {
      setNumeric( val );
   }

   void setNumeric( numeric val ) {
      type( FLC_ITEM_NUM );
      all.ctx.data.number = val;
   }

   /** Creates a range item */
   Item( CoreRange *r )
   {
      setRange( r );
   }

   void setRange( CoreRange *r );

   /** Creates a corestring.
      The given string is copied and stored in the Garbage system.
      It is also bufferized, as the most common usage of this constructor
      is in patterns like Item( "a static string" );
   */
   Item( const String &str );

   /** Creates a CoreString item */
   Item( String *str )
   {
      setString( str );
   }

   /* Creates a String item dependent from a module */
   /*Item( String *str, LiveModule* lm )
   {
      setString( str, lm );
   }*/

   void setString( String *str );
   //void setString( String *str, LiveModule* lm );

   /** Creates an array item */
   Item( CoreArray *array )
   {
      setArray( array );
   }

   void setArray( CoreArray *array );

   /** Creates an object item */
   Item( CoreObject *obj )
   {
      setObject( obj );
   }

   void setObject( CoreObject *obj );

   /** Creates a dictionary item */
   Item( CoreDict *obj )
   {
      setDict( obj );
   }

   void setDict( CoreDict *dict );

   /** Creates a memory buffer. */
   Item( MemBuf *buf )
   {
      setMemBuf( buf );
   }

   void setMemBuf( MemBuf *b );

   Item( GarbageItem *ref )
   {
      setReference( ref );
   }

   /** Creates a reference to another item. */
   void setReference( GarbageItem *ref );

   GarbageItem *asReference() const { return (GarbageItem *) all.ctx.data.ptr.voidp; }

   /** Creates a function item */
   void setFunction( CoreFunc* cf );

   /** Creates a late binding item.
      The late binding is just a CoreString in a live module which is
      resolved into a value by referencing a item in the current
      context (symbol tables) having the given name at runtime.

      Thus, the CoreString representing the late binding symbol name
      lives in the live module that generated this LBind. If
      the module is unloaded, the LBind is invalidated.

      \param lbind The name of the late binding symbol.
      \param val If provided, a future value (future binding).
   */
   void setLBind( String *lbind, GarbageItem *val=0 );

   /** Returns true if this item is a valid LBind.
   */
   bool isLBind() const { return type() == FLC_ITEM_LBIND; }
   bool isFutureBind() const { return isLBind() && all.ctx.data.ptr.extra != 0; }

   /** Return the binding name associate with this LBind item.
   */
   String *asLBind() const { return (String *) all.ctx.data.ptr.voidp; }
   GarbageItem *asFBind() const { return (GarbageItem *) all.ctx.data.ptr.extra; }

   bool isLitLBind() const { return isLBind() && asLBind()->getCharAt(0) == '.'; }


   const Item &asFutureBind() const;
   Item &asFutureBind();

   /** Creates a method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   Item( const Item &data, CallPoint* func )
   {
      setMethod( data, func );
   }

   Item( CoreObject *obj, CoreClass *cls )
   {
      setClassMethod( obj, cls );
   }

   /** Creates a method.
      The method is able to remember if it was called with
      a Function pointer or using an external function.
   */
   void setMethod( const Item &data, CallPoint *func );

   void setClassMethod( CoreObject *obj, CoreClass *cls );

   /** Creates a class item */
   Item( CoreClass *cls )
   {
      setClass( cls );
   }

   void setClass( CoreClass *cls );


   /** Defines this item as a out of band data.
      Out of band data allow out-of-order sequencing in functional programming.
      If an item is out of band, it's type it's still the original one, and
      its value is preserved across function calls and returns; however, the
      out of band data may innescate a meta-level processing of the data
      travelling through the functions.

      In example, returning an out of band NIL value from an xmap mapping
      function will cause xmap to discard the data.
   */
   void setOob() { all.ctx.base.bits.flags |= flagIsOob; }

   /** Clear out of band status of this item.
      \see setOob()
   */
   void resetOob() { all.ctx.base.bits.flags &= ~flagIsOob; }

   /** Sets or clears the out of band status status of this item.
      \param oob true to make this item out of band.
      \see setOob()
   */
   void setOob( bool oob ) {
      if ( oob )
         all.ctx.base.bits.flags |= flagIsOob;
      else
         all.ctx.base.bits.flags &= ~flagIsOob;
   }

   /** Set this item as a user-defined Garbage pointers.
       VM provides GC-control over them.
   */
   void setGCPointer( FalconData *ptr );
   void setGCPointer( GarbagePointer *shell );

   FalconData *asGCPointer() const;
   GarbagePointer *asGCPointerShell() const;

   bool isGCPointer() const { return type() == FLC_ITEM_GCPTR; }

   /** Tells wether this item is out of band.
      \return true if out of band.
      \see oob()
   */
   bool isOob() const { return (all.ctx.base.bits.flags & flagIsOob )== flagIsOob; }

   /** Returns true if this item is an instance of some sort.
      \return true if this is an object, blessed dictionary or bound array.
   */
   bool isComposed() const { return isObject() || isArray() || isDict(); }

   /** Returns the item type*/
   byte type() const { return all.ctx.base.bits.type; }
   
   /** Changes the item type.
   
      Flags are also reset. For example, if this item was OOB before,
      the OOB flag is cleared.
   */
   void type( byte nt ) { all.ctx.base.bits.flags = 0; all.ctx.base.bits.type = nt; }

   /** Returns the content of the item */
   Garbageable *content() const { return all.ctx.data.content; }

   void content( Garbageable *dt ) {
      all.ctx.data.content = dt;
   }

   void copy( const Item &other )
   {
      #ifdef _SPARC32_ITEM_HACK
      register int32 *pthis, *pother;
      pthis = (int32*) this;
      pother = (int32*) &other;
      pthis[0]= pother[0];
      pthis[1]= pother[1];
      pthis[2]= pother[2];
      pthis[3]= pother[3];

      #else
         all = other.all;
      #endif
   }

   /** Tells if this item is callable.
      This function will turn this object into a nil
      if the item referenced a dead module. As this is a pathological
      situation, a const cast is forced.
   */
   bool isCallable() const;

   /** Return true if this is a callable item that is turned into a method when found as property.*/
   bool canBeMethod() const;

   bool isOrdinal() const {
      return type() == FLC_ITEM_INT || type() == FLC_ITEM_NUM;
   }

   bool asBoolean() const
   {
      return all.ctx.data.val32 != 0;
   }

   int64 asInteger() const { return all.ctx.data.val64; }

   numeric asNumeric() const { return all.ctx.data.number; }

   int64 asRangeStart() const { return static_cast<CoreRange*>(all.ctx.data.content)->start(); }
   int64 asRangeEnd()  const { return static_cast<CoreRange*>(all.ctx.data.content)->end(); }
   int64 asRangeStep()  const { return static_cast<CoreRange*>(all.ctx.data.content)->step(); }
   bool asRangeIsOpen()  const { return static_cast<CoreRange*>(all.ctx.data.content)->isOpen(); }
   CoreRange* asRange() const { return static_cast<CoreRange*>(all.ctx.data.content); }

   String *asString() const { return (String *) all.ctx.data.ptr.voidp; }
   LiveModule *asStringModule() const { return (LiveModule *) all.ctx.data.ptr.extra; }
   CoreString *asCoreString() const { return (CoreString *) all.ctx.data.ptr.voidp; }

   DeepItem *asDeepItem() const { return (DeepItem *) all.ctx.data.ptr.voidp; }

   /** Provides a basic CoreString representation of the item.
      Use Falcon::Format for a finer control of item representation.
      \param target a CoreString where the item CoreString representation will be placed.
   */
   void toString( String &target ) const;
   CoreArray *asArray() const { return (CoreArray *) all.ctx.data.ptr.voidp; }
   CoreObject *asObject() const;
   CoreObject *asObjectSafe() const { return (CoreObject *) all.ctx.data.ptr.voidp; }
   CoreDict *asDict() const { return ( CoreDict *) all.ctx.data.ptr.voidp; }
   MemBuf *asMemBuf() const { return ( MemBuf *) all.ctx.data.ptr.voidp; }

   CoreClass* asClass() const { return (CoreClass *) all.ctx.data.ptr.extra; }
   CoreFunc* asFunction() const { return (CoreFunc*) all.ctx.data.ptr.extra; }
   CallPoint* asMethodFunc() const { return (CallPoint*) all.ctx.method; }

   /** Gets the "self" in an item (return the item version). */
   Item asMethodItem() const {
      Item temp = *this;
      temp.type( all.ctx.base.bits.oldType );
      temp = *temp.dereference();
      temp.flagsOn( flagIsMethodic );
      return temp;
   }

   /** Gets the "self" in an item (pass byref version). */
   void getMethodItem( Item &itm ) const {
      itm = *this;
      itm.type( all.ctx.base.bits.oldType );
      itm = *itm.dereference();
      itm.flagsOn( flagIsMethodic );
   }

   /** Turns a method item into its original "self". */
   void deMethod() { type( all.ctx.base.bits.oldType ); }

   CoreClass *asMethodClass() const { return (CoreClass*) all.ctx.data.ptr.extra; }
   CoreObject *asMethodClassOwner() const { return (CoreObject*) all.ctx.data.ptr.voidp; }

   //LiveModule *asModule() const { return all.ctx.data.ptr.m_liveMod; }

   /** Convert current object into an integer.
      This operations is usually done on integers, numeric and CoreStrings.
      It will do nothing meaningfull on other types.
   */
   int64 forceInteger() const ;

   /** Convert current object into an integer.
      This operations is usually done on integers, numeric and CoreStrings.
      It will do nothing meaningfull on other types.

      \note this version will throw a code error if the item is not an ordinal.
   */
   int64 forceIntegerEx() const ;

   /** Convert current object into a numeric.
      This operations is usually done on integers, numeric and CoreStrings.
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
   bool isUnbound() const { return type() == FLC_ITEM_UNB; }
   bool isOfClass( const String &className ) const;

   bool isMethodic() const { return (flags() & flagIsMethodic) != 0; }

   bool isTrue() const;

   Item &operator=( const Item &other ) { copy( other ); return *this; }
   bool operator==( const Item &other ) const { return compare(other) == 0; }
   bool operator!=( const Item &other ) const { return compare(other) != 0; }
   bool operator<(const Item &other) const { return compare( other ) < 0; }
   bool operator<=(const Item &other) const { return compare( other ) <= 0; }
   bool operator>(const Item &other) const { return compare( other ) > 0; }
   bool operator>=(const Item &other) const { return compare( other ) >= 0; }

   bool exactlyEqual( const Item &other ) const;
   
   inline Item *dereference();
   inline const Item *dereference() const;

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
   bool methodize( const Item& self );
   bool methodize( const CoreObject *co )
   {
      return methodize( Item(const_cast<CoreObject *>(co)) );
   }


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
   byte flags() const { return all.ctx.base.bits.flags; }
   void flags( byte b ) { all.ctx.base.bits.flags = b; }
   void flagsOn( byte b ) { all.ctx.base.bits.flags |= b; }
   void flagsOff( byte b ) { all.ctx.base.bits.flags &= ~b; }


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
   bool clone( Item &target ) const;

   /** Return true if the item deep.
      Deep items are the ones that are subject to garbage collecting.
      \return true if the item is deep.
   */
   bool isDeep() const { return type() >= FLC_ITEM_FIRST_DEEP; }

   //====================================================================//

   void add( const Item &operand, Item &result ) const {
      void (*addfunc)( const Item &first, const Item& second, Item &third) =
         (void (*)( const Item &first, const Item& second, Item &third))
         CommOpsDict[type()][co_add];
      addfunc( *this, operand, result );
   }

   void sub( const Item &operand, Item &result ) const {
      void (*func)( const Item &first, const Item& second, Item &third) =
         (void (*)( const Item &first, const Item& second, Item &third))
         CommOpsDict[type()][co_sub];
      func( *this, operand, result );
   }

   void mul( const Item &operand, Item &result ) const {
      void (*func)( const Item &first, const Item& second, Item &third) =
         (void (*)( const Item &first, const Item& second, Item &third))
         CommOpsDict[type()][co_mul];
      func( *this, operand, result );
   }

   void div( const Item &operand, Item &result ) const {
      void (*func)( const Item &first, const Item& second, Item &third) =
         (void (*)( const Item &first, const Item& second, Item &third))
         CommOpsDict[type()][co_div];
      func( *this, operand, result );
   }

   void mod( const Item &operand, Item &result ) const {
      void (*func)( const Item &first, const Item& second, Item &third) =
         (void (*)( const Item &first, const Item& second, Item &third))
         CommOpsDict[type()][co_mod];
      func( *this, operand, result );
   }

   void pow( const Item &operand, Item &result ) const {
      void (*func)( const Item &first, const Item& second, Item &third) =
         (void (*)( const Item &first, const Item& second, Item &third))
         CommOpsDict[type()][co_pow];
      func( *this, operand, result );
   }

   void neg( Item& target ) const {
      void (*func)( const Item &first, Item &tg ) =
         (void (*)( const Item &first, Item &tg ))
         CommOpsDict[type()][co_neg];
      func( *this, target );
   }

   void inc( Item& target ) {
      void (*func)( Item &first, Item &second ) =
         (void (*)( Item &first, Item &second ))
         CommOpsDict[type()][co_inc];
      func( *this, target );
   }

   void dec( Item& target ) {
      void (*func)( Item &first, Item &second ) =
         (void (*)( Item &first, Item &second ))
         CommOpsDict[type()][co_dec];
      func( *this, target );
   }

   void incpost( Item& target ) {
      void (*func)( Item &first, Item &tg ) =
         (void (*)( Item &first, Item &tg ))
         CommOpsDict[type()][co_incpost];
      func( *this, target );
   }

   void decpost( Item& target ) {
      void (*func)( Item &first, Item &tg ) =
         (void (*)( Item &first, Item &tg ))
         CommOpsDict[type()][co_decpost];
      func( *this, target );
   }

   int compare( const Item &operand ) const {
      int (*func)( const Item &first, const Item& second ) =
         (int (*)( const Item &first, const Item& second ))
         CommOpsDict[type()][co_compare];
      return func( *this, operand );
   }

   void getIndex( const Item &idx, Item &result ) const {
      void (*func)( const Item &first, const Item &idx, Item &third) =
         (void (*)( const Item &first, const Item &idx, Item &third))
         CommOpsDict[type()][co_getIndex];
      func( *this, idx, result );
   }

   void setIndex( const Item &idx, const Item &result ) {
      void (*func)( Item &first, const Item &name, const Item &third) =
         (void (*)( Item &first, const Item &name, const Item &third))
         CommOpsDict[type()][co_setIndex];
      func( *this, idx, result );
   }

   void getProperty( const String &property, Item &result ) const {
      void (*func)( const Item &first, const String &property, Item &third) =
         (void (*)( const Item &first, const String &property, Item &third))
         CommOpsDict[type()][co_getProperty];
      func( *this, property, result );
   }

   void setProperty( const String &prop, const Item &result ) {
      void (*func)( Item &first, const String &prop, const Item &third) =
         (void (*)( Item &first, const String &prop, const Item &third))
         CommOpsDict[type()][co_setProperty];
      func( *this, prop, result );
   }

   /** Prepares a call frame that will be called at next VM loop.
      \note You can use vm->execFrame() to execute the prepared frame
      immediately instead of waiting for the loop to complete.
   */
   void readyFrame( VMachine *vm, uint32 paramCount ) const
   {
      void (*func)( const Item &first, VMachine *vm, int paramCount ) =
         (void (*)( const Item &first, VMachine *vm, int paramCount ))
         CommOpsDict[type()][co_call];
      func( *this, vm, paramCount );
   }

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
   GarbageItem( const Item &origin ):
      Garbageable(),
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


class FALCON_DYN_CLASS SafeItem: public Item
{
public:
   SafeItem( const SafeItem &other ) {
      copy( other );
   }

   SafeItem( const Item &other ) {
      copy( other );
   }

   /** Creates a boolean SafeItem. */

   /** Sets this SafeItem as boolean */
   void setBoolean( bool tof )
   {
      type( FLC_ITEM_BOOL );
      all.ctx.data.val32 = tof? 1: 0;
   }

   /** Creates an integer SafeItem */
   SafeItem( int16 val )
   {
      setInteger( (int64) val );
   }

   /** Creates an integer SafeItem */
   SafeItem( uint16 val )
   {
      setInteger( (int64) val );
   }

   /** Creates an integer SafeItem */
   SafeItem( int32 val )
   {
      setInteger( (int64) val );
   }

   /** Creates an integer SafeItem */
   SafeItem( uint32 val )
   {
      setInteger( (int64) val );
   }


   /** Creates an integer SafeItem */
   SafeItem( int64 val )
   {
      setInteger( val );
   }


   /** Creates an integer SafeItem */
   SafeItem( uint64 val )
   {
      setInteger( (int64)val );
   }

   void setInteger( int64 val ) {
      type(FLC_ITEM_INT);
      all.ctx.data.val64 = val;
   }

   /** Creates a numeric SafeItem */
   SafeItem( numeric val )
   {
      setNumeric( val );
   }

   void setNumeric( numeric val ) {
      type( FLC_ITEM_NUM );
      all.ctx.data.number = val;
   }

   SafeItem( byte t, Garbageable *dt );

   SafeItem( CoreRange *r ) { setRange( r ); }
   SafeItem( String *str ) { setString( str ); }
   SafeItem( CoreArray *array ) { setArray( array ); }
   SafeItem( CoreObject *obj ) { setObject( obj ); }
   SafeItem( CoreDict *dict ) { setDict( dict ); }
   SafeItem( MemBuf *b ) { setMemBuf( b ); }
   SafeItem( GarbageItem *r ) { setReference( r ); }
   SafeItem( CoreFunc* cf ) { setFunction( cf ); }
   SafeItem( String *lbind, GarbageItem *val ) { setLBind( lbind, val ); }
   SafeItem( const Item &data, CallPoint *func ) { setMethod( data, func ); }
   SafeItem( CoreObject *obj, CoreClass *cls ) { setClassMethod( obj, cls ); }
   SafeItem( CoreClass *cls ) { setClass( cls ); }
   SafeItem( FalconData *ptr ) { setGCPointer( ptr ); }
   SafeItem( GarbagePointer *shell ) { setGCPointer( shell ); }

   void setRange( CoreRange *r );
   void setString( String *str );
   void setArray( CoreArray *array );
   void setObject( CoreObject *obj );
   void setDict( CoreDict *dict );
   void setMemBuf( MemBuf *b );
   void setReference( GarbageItem *r );
   void setFunction( CoreFunc* cf );
   void setLBind( String *lbind, GarbageItem *val );
   void setMethod( const Item &data, CallPoint *func );
   void setClassMethod( CoreObject *obj, CoreClass *cls );
   void setClass( CoreClass *cls );
   void setGCPointer( FalconData *ptr );
   void setGCPointer( GarbagePointer *shell );
};

}

#endif
/* end of flc_item.h */
