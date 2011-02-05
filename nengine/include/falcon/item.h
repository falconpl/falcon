/*
   FALCON - The Falcon Programming Language.
   FILE: item.h

   Basic Item API.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 06 Jan 2011 13:21:10 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FLC_ITEM_H
#define FLC_ITEM_H

#include <falcon/types.h>
#include <falcon/itemid.h>
#include <falcon/string.h>
#include <falcon/class.h>

namespace Falcon {

class Function;

/** Basic item abstraction.*/
class FALCON_DYN_CLASS Item
{
public:

  struct {
     union {
        int32 val32;
        int64 val64;
        numeric number;

        struct {
           void *pInst;
           CoreClass *pClass;
        } ptr;

     } data;

     union {
        Function *function;
        CoreClass *base;
     } mth;

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
  } content;


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
      content.data.val32 = tof? 1: 0;
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
      content.data.val64 = val;
   }

   /** Creates a numeric item */
   inline Item( numeric val )
   {
      setNumeric( val );
   }

   inline void setNumeric( numeric val ) {
      type( FLC_ITEM_NUM );
      content.data.number = val;
   }

   inline Item( Function* f )
   {
      setFunction(f);
   }

   inline void setFunction( Function* f )
   {
      type( FLC_ITEM_FUNC );
      content.data.ptr.pInst = f;
   }

   inline Item( void* inst, CoreClass* cls ) {
       setDeep( inst, cls );
   }

   inline void setDeep( void* inst, CoreClass* cls )
   {
       type( FLC_ITEM_DEEP );
       content.data.ptr.pInst = inst;
       content.data.ptr.pClass = cls;
   }

   inline void methodize( Function* mthFunc )
   {
       content.mth.function = mthFunc;
       content.base.bits.oldType = content.base.bits.type;
       content.base.bits.type = FLC_ITEM_METHOD;
   }

   inline void methodize( CoreClass* mthClass )
   {
       content.mth.base = mthClass;
       content.base.bits.oldType = content.base.bits.type;
       content.base.bits.type = FLC_ITEM_BASEMETHOD;
   }

   inline void unmethodize()
   {
       content.base.bits.type = content.base.bits.oldType;
   }

   inline void getMethodItem( Item& tgt )
   {
      tgt.content.base.bits.type = content.base.bits.oldType;
      tgt.content.base.bits.flags = 0;
      tgt.content.data.ptr.pInst = content.data.ptr.pInst;
      tgt.content.data.ptr.pClass = content.data.ptr.pClass;
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
   void setOob() { content.base.bits.flags |= flagIsOob; }

   /** Clear out of band status of this item.
      \see setOob()
   */
   void resetOob() { content.base.bits.flags &= ~flagIsOob; }

   /** Sets or clears the out of band status status of this item.
      \param oob true to make this item out of band.
      \see setOob()
   */
   void setOob( bool oob ) {
      if ( oob )
         content.base.bits.flags |= flagIsOob;
      else
         content.base.bits.flags &= ~flagIsOob;
   }


   /** Tells wether this item is out of band.
      \return true if out of band.
      \see oob()
   */
   bool isOob() const { return (content.base.bits.flags & flagIsOob )== flagIsOob; }

   /** Returns the item type*/
   byte type() const { return content.base.bits.type; }
   
   /** Changes the item type.
   
      Flags are also reset. For example, if this item was OOB before,
      the OOB flag is cleared.
   */
   void type( byte nt ) { content.base.bits.flags = 0; content.base.bits.type = nt; }

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
         content = other.content;
      #endif
   }

   /** Assign an item to this item.
    Normally, assign is an item copy, but there are quasi flat items that act
    as if they were flat, but are actually deep. These itmes need to be notified
    about assignments, because they may need to create copies of themselves
    when assigned, or otherwise manipulate their assignments.
    */
   void assign( const Item& other )
   {
       if ( other.type() < FLC_ITEM_DEEP )
       {
           copy( other );
       }
       else
       {
           other.asDeepClass()->assign( other.content.data.ptr.pInst, *this );
       }
   }

   bool asBoolean() const { return content.data.val32 != 0; }
   int64 asInteger() const { return content.data.val64; }
   numeric asNumeric() const { return content.data.number; }
   Function* asFunction() const { return static_cast<Function*>(content.data.ptr.pInst); }

   Function* asMethodFunction() const { return content.mth.function; }
   CoreClass* asMethodClass() const { return content.mth.base; }

   void* asDeepInst() const { return content.data.ptr.pInst; }
   CoreClass* asDeepClass() const { return content.data.ptr.pClass; }

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

   bool isNil() const { return type() == FLC_ITEM_NIL; }
   bool isBoolean() const { return type() == FLC_ITEM_BOOL; }
   bool isInteger() const { return type() == FLC_ITEM_INT; }
   bool isNumeric() const { return type() == FLC_ITEM_NUM; }
   bool isFunction() const { return type() == FLC_ITEM_FUNC; }
   bool isMethod() const { return type() == FLC_ITEM_METHOD; }
   bool isBaseMethod() const { return type() == FLC_ITEM_BASEMETHOD; }
   bool isOrdinal() const { return type() == FLC_ITEM_INT || type() == FLC_ITEM_NUM; }
   bool isDeep() const { return type() == FLC_ITEM_DEEP; }

   bool isTrue() const;

   void toString( String& target ) const;

   Item &operator=( const Item &other ) { copy( other ); return *this; }
   bool operator==( const Item &other ) const { return compare(other) == 0; }
   bool operator!=( const Item &other ) const { return compare(other) != 0; }
   bool operator<(const Item &other) const { return compare( other ) < 0; }
   bool operator<=(const Item &other) const { return compare( other ) <= 0; }
   bool operator>(const Item &other) const { return compare( other ) > 0; }
   bool operator>=(const Item &other) const { return compare( other ) >= 0; }

   bool exactlyEqual( const Item &other ) const;
   int compare( const Item& other ) const;

   /** Flags, used for internal vm needs. */
   byte flags() const { return content.base.bits.flags; }
   void flags( byte b ) { content.base.bits.flags = b; }
   void flagsOn( byte b ) { content.base.bits.flags |= b; }
   void flagsOff( byte b ) { content.base.bits.flags &= ~b; }


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
   bool clone( Item &target ) const
   {
       if ( type() < FLC_ITEM_DEEP )
       {
           target.copy( this );
           return true;
       }
       else
       {
           target.setDeep( asDeepClass()->clone( asDeepInst() ), asDeepClass() );
       }
   }
};

}

#endif
/* end of item.h */
