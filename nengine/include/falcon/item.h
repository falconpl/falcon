/*
   FALCON - The Falcon Programming Language.
   FILE: flc_item.h

   Basic Item API.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Jan 2011 13:21:10 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Basic Item API
*/

#ifndef FLC_ITEM_H
#define FLC_ITEM_H

#include <falcon/types.h>
#include <falcon/itemid.h>
#include <falcon/garbageable.h>
#include <falcon/basealloc.h>
#include <falcon/string.h>

namespace Falcon {

class Function;

/** Basic item abstraction.*/
class FALCON_DYN_CLASS Item: public BaseAlloc
{
public:
   typedef void* CallPoint;

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
   inline Item()
   {
      type( FLC_ITEM_NIL );
   }

   inline void setNil()
   {
      type( FLC_ITEM_NIL );
   }

   inline Item( const Item &other ) {
      copy( other );
   }

   /** Creates a boolean item. */
   inline Item( bool b ) {
      setBoolean( b );
   }

   /** Sets this item as boolean */
   inline void setBoolean( bool tof )
   {
      type( FLC_ITEM_BOOL );
      all.ctx.data.val32 = tof? 1: 0;
   }

   /** Creates an integer item */
   inline Item( int32 val )
   {
      setInteger( (int64) val );
   }

   /** Creates an integer item */
   inline Item( int64 val )
   {
      setInteger( val );
   }

   inline void setInteger( int64 val ) {
      type(FLC_ITEM_INT);
      all.ctx.data.val64 = val;
   }

   /** Creates a numeric item */
   inline Item( numeric val )
   {
      setNumeric( val );
   }

   inline void setNumeric( numeric val ) {
      type( FLC_ITEM_NUM );
      all.ctx.data.number = val;
   }

   inline Item( Function* f )
   {
      setFunction(f);
   }

   inline void setFunction( Function* f )
   {
      type( FLC_ITEM_FUNC );
      all.ctx.data.ptr.voidp = f;
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


   /** Tells wether this item is out of band.
      \return true if out of band.
      \see oob()
   */
   bool isOob() const { return (all.ctx.base.bits.flags & flagIsOob )== flagIsOob; }

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


   bool isOrdinal() const {
      return type() == FLC_ITEM_INT || type() == FLC_ITEM_NUM;
   }

   bool asBoolean() const
   {
      return all.ctx.data.val32 != 0;
   }

   int64 asInteger() const { return all.ctx.data.val64; }
   numeric asNumeric() const { return all.ctx.data.number; }

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

   String* asSymbolName() const { return (String*) all.ctx.data.ptr.extra; }
   Item* asSymbolValue() const { return (Item*) all.ctx.data.ptr.voidp; }

   bool isNil() const { return type() == FLC_ITEM_NIL; }
   bool isBoolean() const { return type() == FLC_ITEM_BOOL; }
   bool isInteger() const { return type() == FLC_ITEM_INT; }
   bool isNumeric() const { return type() == FLC_ITEM_NUM; }
   bool isScalar() const { return type() == FLC_ITEM_INT || type() == FLC_ITEM_NUM; }

   bool isObject() const { return type() == FLC_ITEM_OBJECT; }
   bool isSymbol() const { return type() == FLC_ITEM_SYMBOL; }
   bool isFunction() const { return type() == FLC_ITEM_FUNC; }
   Function* asFunction() const { return (Function*) all.ctx.data.ptr.voidp; }

   bool isTrue() const;

   virtual void toString( String& target ) const;

   Item &operator=( const Item &other ) { copy( other ); return *this; }
   /*
   bool operator==( const Item &other ) const { return compare(other) == 0; }
   bool operator!=( const Item &other ) const { return compare(other) != 0; }
   bool operator<(const Item &other) const { return compare( other ) < 0; }
   bool operator<=(const Item &other) const { return compare( other ) <= 0; }
   bool operator>(const Item &other) const { return compare( other ) > 0; }
   bool operator>=(const Item &other) const { return compare( other ) >= 0; }
    */
   bool exactlyEqual( const Item &other ) const;

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
   //bool methodize( const Item& self );
   /*bool methodize( const CoreObject *co )
   {
      return methodize( Item(const_cast<CoreObject *>(co)) );
   }*/


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
};

}

#endif
/* end of item.h */
