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

#include <falcon/atomic.h>

namespace re2 {
   class RE2;
}

namespace Falcon {

class Function;
class ItemArray;
class ItemDict;

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

        struct {
           void* pOpaque;
           const char* pOpaqueName;
        } opaque;

     } data;

     union {
        Function *function;
        Class *base;
        Item* ref;
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

  mutable atomic_int lockId;


#ifdef _MSC_VER
	#if _MSC_VER < 1299
	#define flagLiteral 0x01
	#define flagIsGarbage 0x02
	#define flagIsOob 0x04
	#define flagLast 0x08
	#define flagContinue 0x10
	#define flagBreak 0x20
	#else
	   static const byte flagLiteral = 0x01;
	   static const byte flagIsGarbage = 0x02;
	   static const byte flagIsOob = 0x04;
	   static const byte flagDoubt = 0x08;
	   static const byte flagContinue = 0x10;
	   static const byte flagBreak = 0x20;
	#endif
#else
   static const byte flagLiteral = 0x01;
   static const byte flagIsGarbage = 0x02;
   static const byte flagIsOob = 0x04;
   static const byte flagDoubt = 0x08;
   static const byte flagContinue = 0x10;
   static const byte flagBreak = 0x20;
#endif

   static Class* m_funcClass;
   static Class* m_stringClass;
   static Class* m_dictClass;
   static Class* m_arrayClass;
      
public:
   /** Initializes the item system.
    Called by the engine during startup.
    */
   static void init( Engine* eng );

      
   inline Item()
   {
      lockId = 0;
      type( FLC_ITEM_NIL );
   }

   inline void setNil()
   {
      type( FLC_ITEM_NIL );
   }

   inline Item( const Item &other )
   {
      lockId = 0;
      other.lock();
      copy( other );
      other.unlock();
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
      lockId = 0;
      setString(str);
   }

   Item( const char* name, const void* opaque )
   {
      lockId = 0;
      setOpaque(name, opaque);
   }

   Item( const Symbol* sym )
   {
      setSymbol( sym );
   }

   Item& setOpaque( const char* name, const void* opaque ) {
      type( FLC_ITEM_OPAQUE );
      content.data.opaque.pOpaqueName = name;
      content.data.opaque.pOpaque = (void*) opaque;
      return *this;
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
      lockId = 0;
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
      lockId = 0;
      setString(str);
   }

   /** Sets this item to a String item.
    \param str A C string that is taken as-is and converted to a Falcon String.
    \see Item( char* )
    */
   Item& setString( const char* str );

   /** Sets this item to a String item.
    \param str A wchar_t string that is taken as-is and converted to a Falcon String.
    \see Item( wchar_t* )
    */
   Item& setString( const wchar_t* str );

   /** Sets this item to a String item.
    \param str A Falcon string that will be copied.
    \see Item( const String& )
    */
   Item& setString( const String& str );

   Item& setSymbol( const Symbol* sym );

   /** Creates a boolean item. */
   explicit inline Item( bool b ) {
      lockId = 0;
      setBoolean( b );
   }

   /** Sets this item as boolean */
   inline Item& setBoolean( bool tof )
   {
      lockId = 0;
      type( FLC_ITEM_BOOL );
      content.data.val32 = tof? 1: 0;
      return *this;
   }

   /** Creates an integer item */
   inline Item( int32 val )
   {
      lockId = 0;
      setInteger( (int64) val );
   }

   /** Creates an integer item */
   inline Item( int64 val )
   {
      lockId = 0;
      setInteger( val );
   }

   inline Item& setInteger( int64 val ) {
      type(FLC_ITEM_INT);
      content.data.val64 = val;
      return *this;
   }

   /** Creates a numeric item */
   inline Item( numeric val )
   {
      lockId = 0;
      setNumeric( val );
   }

   inline Item& setNumeric( numeric val ) {
      type( FLC_ITEM_NUM );
      content.data.number = val;
      return *this;
   }

   inline Item( Function* f )
   {
      lockId = 0;
      setFunction(f);
   }

   inline Item& setFunction( Function* f )
   {
      type( FLC_CLASS_ID_FUNC );
      content.data.ptr.pInst = f;
      content.data.ptr.pClass = Item::m_funcClass;
      return *this;
   }

   inline Item( const Class* cls, const void* inst ) {
      lockId = 0;
       setUser( cls, inst );
   }


   inline Item& setUser( const Class* cls, const void* inst )
   {
       type( (byte) cls->typeID() );
       content.data.ptr.pInst = (void*) inst;
       content.data.ptr.pClass = (Class*) cls;
       content.base.bits.flags &= ~flagIsGarbage;
       return *this;
   }

   inline Item( GCToken* token )
   {
      lockId = 0;
      setUser( token );
   }
   
   inline Item& setUser( GCToken* token )
   {
       type( (byte) token->cls()->typeID() ); // normally
       content.data.ptr.pClass = token->cls();
       content.data.ptr.pInst = token->data();
       content.base.bits.flags |= flagIsGarbage;
       return *this;
   }


   inline Item& methodize( Function* mthFunc )
   {
       content.mth.function = mthFunc;
       content.base.bits.oldType = content.base.bits.type;
       content.base.bits.type = FLC_ITEM_METHOD;
       return *this;
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

      The copy marker is cleared as well.
   */
   void type( byte nt ) {
      content.base.bits.flags = 0;
      content.base.bits.type = nt;
   }

   /**
    * True if the instance in this item comes from the garbage collector.
    *
    * Meaningful only if the item is a user item (isUser() is true).
    */
   bool isGarbage() {
      return (content.base.bits.flags & flagIsGarbage) != 0;
   }

   /**
    * Use with caution.
    */
   void setGarbage() {
      content.base.bits.flags |= flagIsGarbage;
   }


   /** Copies the full contents of another item.
      \param other The copied item.

    This performs a full flat copy of the original item.

    \note This doesn't set the copy marker; the copy marker is meant to determine
    if the VM has copied an item a script level while basic copy operations
    may not have this semantic at lower level. For example, one may make a
    temporary flat copy of a stack item without willing to notify that to the
    VM.
    */
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
    This operation performs a three-way full interlocked copy of the item.
    To be used when both items might be in a global space.
    */
   void copyInterlocked( const Item& other  )
   {
      Item temp;
      other.lock();
      temp.copy(other);
      other.unlock();

      lock();
      copy(temp);
      unlock();
   }

   /**
    * Assign without interlocking the remote item.
    * \param local The local item from which we do an unlocked copy.
    *
    * This method only locks this item; this is ok
    * if the source is in the local context data stack.
    */
   void copyFromLocal( const Item& local  )
   {
      lock();
      copy(local);
      unlock();
   }

   /**
    * Perform an assignment from a remote locked item to a local copy.
    * \param remote The remote, non local item from which we do locked copy.
    *
    * This method only locks the remote item; this is ok
    * if this item is in the local context data stack.
    */
   void copyFromRemote( const Item& remote )
   {
      remote.lock();
      copy(remote);
      remote.unlock();
   }


   bool asBoolean() const { return content.data.val32 != 0; }
   int64 asInteger() const { return content.data.val64; }
   int64 asOrdinal() const { return isInteger() ? content.data.val64 : (int64) content.data.number; }
   numeric asNumeric() const { return content.data.number; }
   Function* asFunction() const { return static_cast<Function*>(content.data.ptr.pInst); }

   Function* asMethodFunction() const { return content.mth.function; }
   Class* asMethodClass() const { return content.mth.base; }

   void* asInst() const { return content.data.ptr.pInst; }
   /** Synonymous of asInst() */
   void* asUser() const { return content.data.ptr.pInst; }

   /** Shortcut to get the parent data of an instance.
    * @param parent the parent base class.
    * @return 0 if this item is not compatible with the given parent class, a valid
    *         instance for the parent class on success.
    *
    * This method returns the base class data stored in a subclass-instance item.
    * It's equivalent to call asClassInst() to get the class and instance stored
    * in this item, and then use Class::getParentData to get the instance data
    * as the parent knows it.
    */
   void* asParentInst( const Class* parent ) const;

   /** Shortcut to get the parent data of an instance.
    * @param parent the parent base class.
    * @return 0 if this item is not compatible with the given parent class, a valid
    *         instance for the parent class on success.
    *
    * This method returns the base class data stored in a subclass-instance item.
    * It's equivalent to call forceParentInst() to get the class and instance stored
    * in this item, and then use Class::getParentData to get the instance data
    * as the parent knows it.
    *
    * @note The difference with respect to asParentInst() is that this version works
    * with flat data. asParentInst() should be used with items known to be non-flat
    * data only.
    */
   void* forceParentInst( const Class* parent );


   Class* asClass() const { return content.data.ptr.pClass; }

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
   bool isFunction() const { return type() == FLC_CLASS_ID_FUNC; }
   bool isMethod() const { return type() == FLC_ITEM_METHOD; }
   bool isOrdinal() const { return type() == FLC_ITEM_INT || type() == FLC_ITEM_NUM; }
   bool isCallable() const;
   bool isTreeStep() const  { return type() == FLC_CLASS_ID_TREESTEP; }
   bool isOpaque() const { return type() == FLC_ITEM_OPAQUE; }
   bool isSymbol() const { return type() == FLC_CLASS_ID_SYMBOL; }
   
   bool isUser() const { return type() >= FLC_ITEM_USER; }
   bool isClass() const { return type() == FLC_CLASS_ID_CLASS; }
   
   /** A stub for now */
   bool isMemBuf() const { return false; } 
   bool isString() const {
      return (type() == FLC_CLASS_ID_STRING);
   }

   bool isRE() const {
      return (type() == FLC_CLASS_ID_RE );
   }

   bool isRange() const {
      return (type() == FLC_CLASS_ID_RANGE);
   }

   bool isArray() const {
      return (type() == FLC_CLASS_ID_ARRAY);
   }

   bool isDict() const {
      return (type() == FLC_CLASS_ID_DICT);
   }

   
   bool isTrue() const;
   
   /** Turns an item into its non-user class form.
    \return True if the item can be de-usered.
    
    Suppose a flat item has been turned into a pair of user-data entities
    through a forceClassInst() call. It's sometimes useful to perform the
    reverse operation, re-obtaining a flat item out of this pair.
    \note Can be called only on items that have been already tested as being
    FLC_ITEM_USER.
    */
   bool deuser()
   {
      fassert2(type() >= FLC_ITEM_USER, 
            "Item::deuser() must be called only on FLC_ITEM_USER items.");
      int id = (int)asClass()->typeID();
      if( id < FLC_ITEM_COUNT )
      {
         *this = *static_cast<Item*>(asInst());
         return true;
      }
      else
      {
         return false;
      }
   }
   
   /** Turns the item into a string.
    This method turns the item into a string for a minimal external representation.

    In case the item is deep, it will use the Class::describe() member to obtain
    a representation.
    */
   void describe( String& target, int depth = 3, int maxLen = 60 ) const;

   String describe( int depth = 3, int maxLen = 60) const {
      String t;
      describe(t, depth, maxLen);
      return t;
   }

   /** Operator version of copy.
    \note This doesn't set the copy marker. Use assign() for that.
    */
   Item &operator=( const Item &other ) { copy( other ); return *this; }
   bool operator==( const Item &other ) const { return compare(other) == 0; }
   bool operator!=( const Item &other ) const { return compare(other) != 0; }
   bool operator<(const Item &other) const { return compare( other ) < 0; }
   bool operator<=(const Item &other) const { return compare( other ) <= 0; }
   bool operator>(const Item &other) const { return compare( other ) > 0; }
   bool operator>=(const Item &other) const { return compare( other ) >= 0; }

   bool exactlyEqual( const Item &other ) const;
   int64 compare( const Item& other ) const;

   /** Flags, used for internal vm needs. */
   byte flags() const { return content.base.bits.flags; }
   void flags( byte b ) { content.base.bits.flags = b; }
   void flagsOn( byte b ) { content.base.bits.flags |= b; }
   void flagsOff( byte b ) { content.base.bits.flags &= ~b; }

   bool isDoubt() const { return (content.base.bits.flags & flagDoubt ) != 0; }
   bool isBreak() const { return (content.base.bits.flags & flagBreak ) != 0; }
   bool isContinue() const { return (content.base.bits.flags & flagContinue ) != 0; }

   Item& setDoubt() { content.base.bits.flags |= flagDoubt; return *this;}
   Item& clearDoubt() { content.base.bits.flags &= ~flagDoubt; return *this;}
   Item& setBreak() { content.base.bits.flags |= flagBreak; return *this;}
   Item& setContinue() { content.base.bits.flags |= flagContinue; return *this;}

   /** GC Mark the item -- if necessary. 
    Provided this is an item in need of GC marking, this method asks the
    item class to mark the item. Otherwise, the operation is no-op.
    
    */
   
   void gcMark( uint32 mark ) const
   {
      if( isUser() )
      {
         asClass()->gcMarkInstance( asInst(), mark );
      }
   }

   /** Clone the item.
      If the item is not cloneable, the method returns false. Is up to the caller to
      raise an appropriate error if that's the case.
      The VM parameter may be zero; in that case, returned items will not be stored in
      any garbage collector.

      Reference items are de-referenced; if cloning a reference, the caller will obtain
      a clone of the target item, not a clone of the reference.

      Also, in that case, the returned item will be free of reference.

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
      if( isUser() )
      {
         cls = asClass();
         udata = asInst();
         return true;
      }
      return false;
   }

   ItemArray* asArray() const
   {
      return static_cast<ItemArray*>(asInst());
   }
   
   String* asString() const
   {      
      return static_cast<String*>(asInst());
   }

   re2::RE2* asRE() const
   {
      return static_cast<re2::RE2*>(asInst());
   }

   const Symbol* asSymbol() const
   {
      return static_cast<Symbol*>(asInst());
   }


   ItemDict* asDict() const
   {
      return static_cast<ItemDict*>(asInst());
   }
   
   Item* asReference() const
   {
      return content.mth.ref;
   }

   void* asOpaque() const {
      return content.data.opaque.pOpaque;
   }

   const char* asOpaqueName() {
      return content.data.opaque.pOpaqueName;
   }

   /** Checks if this item is an instance of the given class, or of a child class.
    * @param cls A Falcon::Class instance.
    * @return true if the check succeeds.
    *
    * This methods check if the given item is an instance (via isUser()), and then if
    * the class of the instance is cls or a child of cls.
    *
    * This method will not work for flat objects (integers etc) unless the item
    * has been previusly unflattened.
    */
   bool isInstanceOf( const Class* cls ) const
   {
      return isUser() && asClass()->isDerivedFrom(cls);
   }

   
   /** Gets the class and instance from any item.
     \param cls The class of this deep or user item.
     \param udata The user data associated with this item or 0 if this item is flat.

    In case of flat items, a slower call is made to identify the
    base handler class, and that is returned. In that case, the udata*
    is set to "this" pointer (items always start with the data, then
    an optional method pointer, and finally the type and description portion,
    so this is consistent even when the handler class is expecting a something
    that's not necessarily an Item*).
   */
   void forceClassInst( Class*& cls, void*& udata ) const
   {
      if ( type() >= FLC_ITEM_USER )
      {
         cls = asClass();
         udata = asInst();
      }
      else {         
         cls = Engine::instance()->getTypeClass(type());
         udata = (void*)this;
      }
   }
  
   //======================================================
   // Utilities
   //

   int64 len() const;
   void swap( Item& other ) { Item temp = *this; *this = other; other = temp; }

   void lock() const {
      while(!atomicCAS(lockId, 0, 1)) {}
   }

   void unlock() const {
      atomicSet(lockId, 0);
   }

   /**  Typedefined instance retriever.
    *
    * This method works like asParentInst(), extracting the correct instance for the given class
    * out of the instance in this item, and adding a useful typecast.
    */
   template< class T_ >
   T_* castInst(const Class* baseClass) const
   {
      fassert(isUser());

      Class* cls = 0;
      void* data = 0;
      this->asClassInst(cls,data);
      T_* item = static_cast<T_*>(cls->getParentData(baseClass, data));
      return item;
   }
};

}

#endif
/* end of item.h */
