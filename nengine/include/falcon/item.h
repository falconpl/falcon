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
#include <falcon/gctoken.h>
#include <falcon/collector.h>
#include <falcon/engine.h>

namespace Falcon {

class Function;
class ItemArray;

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
           Class *pClass;
        } ptr;

        GCToken* pToken;

     } data;

     union {
        Function *function;
        Class *base;
        int32 ruleTop;
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

   /** Creates a String item.
    \param str A C string that is taken as-is and converted to a Falcon String.
    
    A Falcon String is created as static (the input \b str is considered to stay
    valid throut the lifetime of the program), and then the String is sent to
    the garbage collector.

    In case the input data is transient, use the Item( const String& ) constructor
    instead, throught the following idiom:

    @code
      char *cptr = ret_transient();
      Item item( String( cptr, -1 ) );
    @endcode

    \note The operation is Encoding neuter.
    */
   Item( const char* str )
   {
      setString(str);
   }

   /** Creates a String item.
    \param str A wchar_t string that is taken as-is and converted to a Falcon String.

     A Falcon String is created as static (the input \b str is considered to stay
    valid throut the lifetime of the program), and then the String is sent to
    the garbage collector.

    In case the input data is transient, use the Item( const String& ) constructor
    instead, throught the following idiom:

    @code
      wchar_t *cptr = ret_transient();
      Item item( String( cptr, -1 ) );
    @endcode

    \note The operation is Encoding neuter (the input is considered UDF).
    */
   Item( const wchar_t* str )
   {
      setString(str);
   }

   /** Creates a String item.
    \param str A Falcon string that will be copied.

    The deep contents of the string are copied only if the string is
    buffered; static strings (strings created out of static char* or wchar_t*
    data) are not deep-copied.

    The created string is subject to garbage collecting.
    */
   Item( const String& str )
   {
      setString(str);
   }

   /** Creates a String item, adopting an existing string.
    \param str A pointer to a Falcon String that must be adopted.

    The string is taken as-is and stored for garbage in the Falcon engine.
    This constructor is useful when a String* has been created for another reason,
    and then must be sent to the engine as a Falcon item. In this way, the
    copy (and eventual deep-copy) of the already created string is avoided.
    
    */
   Item( String* str )
   {
      setString(str);
   }

   /** Sets this item to a String item.
    \param str A C string that is taken as-is and converted to a Falcon String.
    \see Item( char* )
    */
   void setString( const char* str );

   /** Sets this item to a String item.
    \param str A wchar_t string that is taken as-is and converted to a Falcon String.
    \see Item( wchar_t* )
    */
   void setString( const wchar_t* str );

   /** Sets this item to a String item.
    \param str A Falcon string that will be copied.
    \see Item( const String& )
    */
   void setString( const String& str );

   /** Sets this item to a String item.
    \param str A pointer to a Falcon String that must be adopted.
    \see Item( String* )
    */
   void setString( String* str );

   /** Creates a boolean item. */
   explicit inline Item( bool b ) {
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

   inline Item( Class* cls, void* inst ) {
       setUser( cls, inst );
   }

   inline void setUser( Class* cls, void* inst )
   {
       type( FLC_ITEM_USER );
       content.data.ptr.pInst = inst;
       content.data.ptr.pClass = cls;
   }

   inline Item( GCToken* token )
   {
      setDeep( token );
   }
   
   inline void setDeep( GCToken* token )
   {
       type( FLC_ITEM_DEEP );
       content.data.pToken = token;
   }

   inline void methodize( Function* mthFunc )
   {
       content.mth.function = mthFunc;
       content.base.bits.oldType = content.base.bits.type;
       content.base.bits.type = FLC_ITEM_METHOD;
   }

   inline void methodize( Class* mthClass )
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

   /** Turn this item in to an array (deep).
    \param array The array handled to the vm.
    
    This method turns the item into an array with deep semantics (i.e.
    controlled by the garbage collector.

    For user-semantic (i.e. user-controlled) you must use directly the
    setUser() method.
    */
   void setArray( ItemArray* array );

   /** Turn this item in to a dictionary (deep).
    \param dict The dictionary handled to the vm
    
    This method turns the item into an dictionary with deep semantics (i.e.
    controlled by the garbage collector.

    For user-semantic (i.e. user-controlled) you must use directly the
    setUser() method.
    */
   //void setDict( ItemDictionary* dict );

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

   /** Toggle out of band status of this item.
      \see setOob() resetOob()
   */
   void xorOob() { content.base.bits.flags ^= flagIsOob; }

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
       switch ( other.type() )
       {
       case FLC_ITEM_DEEP:
       {
          register GCToken* gt = other.content.data.pToken;
          if( gt->cls()->isFlat() )
          {
             register void* value = gt->cls()->assign( gt->data() );
             if ( value == gt->data() )
             {
                setDeep( gt );
             }
             else
             {
                setDeep( gt->collector()->store( gt->cls(), value ) );
             }
          }
          else
          {
             setDeep( gt );
          }
       }
      break;

      case FLC_ITEM_USER:
         if( other.content.data.pToken->cls()->isFlat() )
         {
            setUser( other.asUserClass(), other.asUserClass()->assign( other.asUserInst() ) );
         }
         else
         {
            setUser( other.asUserClass(), other.asUserInst() );
         }
         break;

       default:
           copy( other );
       }
   }

   bool asBoolean() const { return content.data.val32 != 0; }
   int64 asInteger() const { return content.data.val64; }
   numeric asNumeric() const { return content.data.number; }
   Function* asFunction() const { return static_cast<Function*>(content.data.ptr.pInst); }

   Function* asMethodFunction() const { return content.mth.function; }
   Class* asMethodClass() const { return content.mth.base; }

   void* asUserInst() const { return content.data.ptr.pInst; }
   Class* asUserClass() const { return content.data.ptr.pClass; }

   void* asDeepInst() const { return content.data.pToken->data(); }
   Class* asDeepClass() const { return content.data.pToken->cls(); }
   GCToken* asDeep() const { return content.data.pToken; }

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

   bool isString() const {
      return
        (type() == FLC_ITEM_DEEP && asDeepClass()->typeID() == FLC_CLASS_ID_STRING)
        || (type() == FLC_ITEM_USER && asUserClass()->typeID() == FLC_CLASS_ID_STRING);
   }

   bool isArray() const {
      return
        (type() == FLC_ITEM_DEEP && asDeepClass()->typeID() == FLC_CLASS_ID_ARRAY)
        || (type() == FLC_ITEM_USER && asUserClass()->typeID() == FLC_CLASS_ID_ARRAY);
   }

   bool isDict() const {
      return
        (type() == FLC_ITEM_DEEP && asDeepClass()->typeID() == FLC_CLASS_ID_DICT)
        || (type() == FLC_ITEM_USER && asUserClass()->typeID() == FLC_CLASS_ID_DICT);
   }

   bool isTrue() const;

   /** Turns the item into a string.
    This method turns the item into a string for a minimal external representation.

    In case the item is deep, it will use the Class::describe() member to obtain
    a representation.
    */
   void describe( String& target ) const;

   String describe() const { String t; describe(t); return t; }

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
   bool clone( Item &target ) const;


   /** Gets the class and the instance from a deep item.
    \param cls The class of this deep or user item.
    \param udata The user data associated with this item.
    \return false if the item is not deep or user.

    This method simplifies the detection of deep items and the
    extraction of their vital elements (user data and handling class).

    This method fails if the object is flat. A more generic function is
    forceClassInst() that will return the handler class also for simple items;
    however forceClassInst() is slower.
    */
   bool asClassInst( Class*& cls, void*& udata ) const
   {
      switch( type() )
      {
      case FLC_ITEM_DEEP:
         cls = asDeepClass();
         udata = asDeepInst();
         return true;

      case FLC_ITEM_USER:
         cls = asUserClass();
         udata = asUserInst();
         return true;
      }
      return false;
   }

   /** Gets the only the instance from a deep item.
    \return The deep or user instance of an item.
    */
   void* asInst() const
   {
      switch( type() )
      {
      case FLC_ITEM_DEEP:
         return asDeepInst();
      case FLC_ITEM_USER:
         return asUserInst();
      }
      return 0;
   }


   
   /** Gets the class and instance from any item.
     \param cls The class of this deep or user item.
     \param udata The user data associated with this item.

    In case of flat items, a slower call is made to identify the
    base handler class, and that is returned. In that case, the udata*
    is set to "this" pointer (items always start with the data, then
    an optional method pointer, and finally the type and description portion,
    so this is consistent even when the handler class is expecting a something
    that's not necessarily an Item*).
   */
   void forceClassInst( Class*& cls, void*& udata )
   {
      switch( type() )
      {
      case FLC_ITEM_DEEP:
         cls = asDeepClass();
         udata = asDeepInst();
         break;

      case FLC_ITEM_USER:
         cls = asUserClass();
         udata = asUserInst();
         break;

      default:
         cls = Engine::instance()->getTypeClass(type());
         udata = this;
      }
   }

   //===================================================================
   // Is string?
   //
   
   bool isString( String*& str )
   {
      Class* cls;
      if ( asClassInst( cls, (void*&)str ) )
      {
         return cls->typeID() == FLC_CLASS_ID_STRING;
      }
      return false;
   }

   String* asString()
   {
      Class* cls;
      void* udata;
      if ( asClassInst( cls, udata ) )
      {
         if( cls->typeID() == FLC_CLASS_ID_STRING )
            return static_cast<String*>(udata);
      }

      return 0;
   }

};

}

#endif
/* end of item.h */
