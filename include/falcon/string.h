/*
   FALCON - The Falcon Programming Language.
   FILE: string.h
   $Id: string.h,v 1.20 2007/08/19 09:46:44 jonnymind Exp $

   Core falcon string representation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven nov 19 2004
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
   Core falcon string representation
*/

#ifndef flc_string_H
#define flc_string_H

#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <stdlib.h>

#define FALCON_STRING_ALLOCATION_BLOCK 32

namespace Falcon {

class Stream;
class VMachine;

/** Core falcon string representation.
   This is a string as seen by the VM and its fellows (module loader, module and so on).
   As times goes by, it will substitute ever c_string and Hstring ( stl++ string + falcon
   allocator), so it must be small and efficient, and support different allocation schemes
   seamlessly.

   Strings are polymorphic; that is, they always occupy the same amount of space,
   independently of the kind (being the string deep data pointed by a pointer in
   the class), so that the type of a string may be changed without the need to
   reallocate it. So, if/when more than one item are pointing to the core string object,
   and the core string changes, the new status is immediately reflected to all the pointing
   items. This wastes a little bit of allocation space on every string, but speeds up the
   engine notably.

*/

class String;

/** Core string utility namespace.
   Core strings cannot be virtual classes. This is because core strings
   may be polimorphic; some operations may change the very nature of
   core strings. In example, read-only static strings will be turned to
   memory buffers in case of write operations. Chunked strings may be
   turned into sequential buffers, and the other way around.
   Re-creating them with a new() operator would not provide a viable solution
   as all the other data must stay the same.

   For this reason, a manager class hyerarcy is provided. Every core string
   has a manager, (which is actually a pointer to a pure-virtual class with
   no data), and every operation on a string is an inline call to the manager
   class. So, the string virtuality is externalized in the manager.

   With a "standard" compiler optimization this is 100% equivalent to have a
   pointer to a table of function, but actually faster as the offset of the
   virtual table is calculated at compile time in every situation.

*/

namespace csh {

/** Type of core string.
   As the type of the core string is extracted by the manager,
   this is actually stored in the manager.
*/
typedef enum
{
   cs_static,
   cs_buffer,
   cs_static16,
   cs_buffer16,
   cs_static32,
   cs_buffer32
} t_type;

/** Invalid position for core strings. */
const uint32 npos = 0xFFFFFFFF;

/** Base corestring manager class.
   This is actually an interface that must be implemented by all the core string managers.
*/
class FALCON_DYN_CLASS Base
{
public:
   virtual t_type type() const =0;
   virtual uint32 charSize() const = 0;
   virtual uint32 length( const String *str ) const =0;
   virtual uint32 getCharAt( const String *str, uint32 pos ) const =0;

   virtual void setCharAt( String *str, uint32 pos, uint32 chr ) const =0;
   virtual void subString( const String *str, int32 start, int32 end, String *target ) const =0;
   /** Finds a substring in a string, and eventually returns npos if not found. */
   virtual uint32 find( const String *str, const String *element, uint32 start =0, uint32 end = npos) const = 0;
   virtual uint32 rfind( const String *str, const String *element, uint32 start =0, uint32 end = npos) const = 0;
   virtual void insert( String *str, uint32 pos, uint32 len, const String *source ) const =0;
   virtual bool change( String *str, uint32 start, uint32 end, const String *source ) const =0;
   virtual void remove( String *str, uint32 pos, uint32 len ) const =0;
   virtual String *clone( const String *str ) const =0;
   virtual void destroy( String *str ) const =0;

   virtual void bufferize( String *str ) const =0;
   virtual void bufferize( String *str, const String *strOrig ) const =0;
   virtual void reserve( String *str, uint32 size, bool relative = false, bool block = false ) const = 0;
   virtual void shrink( String *str ) const = 0;

   virtual const Base *bufferedManipulator() const =0;
};

/** Byte orientet base class.
   This is still an abstract class, but it provides minimal behavior for byte oriented
   strings (ascii or system specific).
*/
class FALCON_DYN_CLASS Byte: public Base
{
public:
   virtual uint32 length( const String *str ) const;
   virtual uint32 getCharAt( const String *str, uint32 pos ) const;
   virtual void subString( const String *str, int32 start, int32 end, String *target ) const;
   virtual bool change( String *str, uint32 pos, uint32 end, const String *source ) const;
   virtual String *clone( const String *str ) const;
   virtual uint32 find( const String *str, const String *element, uint32 start =0, uint32 end = 0) const;
   virtual uint32 rfind( const String *str, const String *element, uint32 start =0, uint32 end = 0) const;
   virtual void remove( String *str, uint32 pos, uint32 len ) const;

   virtual void bufferize( String *str ) const;
   virtual void bufferize( String *str, const String *strOrig ) const;
   virtual void reserve( String *str, uint32 size, bool relative = false, bool block = false ) const;
   virtual const Base *bufferedManipulator() const { return this; }
};


/** Static byte oriented string manager.
   Useful to instantiante and manage strings whose content is byte oriented and whose size is
   known in advance; in example, symbol names in the Falcon module are easily managed with this class.

   Every write operation on strings managed by this class will cause its manager to be changed
   into the Buffer class.
*/
class FALCON_DYN_CLASS Static: public Byte
{
public:
   virtual t_type type() const { return cs_static; }
   virtual uint32 charSize() const { return 1; }

   virtual void setCharAt( String *str, uint32 pos, uint32 chr ) const;
   virtual void insert( String *str, uint32 pos, uint32 len, const String *source ) const;
   virtual void remove( String *str, uint32 pos, uint32 len ) const;
   virtual void destroy( String *str ) const;

   virtual void reserve( String *str, uint32 size, bool relative = false, bool block = false ) const;
   virtual void shrink( String *str ) const;
   virtual const Base *bufferedManipulator() const;
};


/** Variable size byte oriented string.
   This class manages a variable size strings that are stored in one region of memory.
   Strings may or may not be zero terminated (in case of need, the zero after the length of
   the string is checked, and if not present, added). This is actually a useful class to
   store C strings created on the fly from memory buffer; the requirement is that the
   memory stored in the managed class is created with the Falcon::memAlloc() function
   (as it will be freed with Falcon::memFree() and reallocated with Falcon::memRealloc() ).
*/
class FALCON_DYN_CLASS Buffer: public Byte
{
public:
   virtual t_type type() const { return cs_buffer; }
   virtual uint32 charSize() const { return 1; }

   virtual void setCharAt( String *str, uint32 pos, uint32 chr ) const;
   virtual void insert( String *str, uint32 pos, uint32 len, const String *source ) const;
   virtual void destroy( String *str ) const;
   virtual void reserve( String *str, uint32 size, bool relative = false, bool block = false ) const;
   virtual void shrink( String *str ) const;

};

class FALCON_DYN_CLASS Static16: public Static
{
public:
   virtual uint32 charSize() const  { return 2; }
   virtual uint32 length( const String *str ) const;
   virtual uint32 getCharAt( const String *str, uint32 pos ) const;
   virtual void setCharAt( String *str, uint32 pos, uint32 chr ) const;
   virtual void insert( String *str, uint32 pos, uint32 len, const String *source ) const;
   virtual void remove( String *str, uint32 pos, uint32 len ) const;
   virtual void reserve( String *str, uint32 size, bool relative = false, bool block = false ) const;
   virtual const Base *bufferedManipulator() const;
};

class FALCON_DYN_CLASS Static32: public Static16
{
public:
   virtual uint32 charSize() const { return 4; }
   virtual uint32 length( const String *str ) const;
   virtual uint32 getCharAt( const String *str, uint32 pos ) const;
   virtual void setCharAt( String *str, uint32 pos, uint32 chr ) const;
   virtual void insert( String *str, uint32 pos, uint32 len, const String *source ) const;
   virtual void remove( String *str, uint32 pos, uint32 len ) const;
   virtual void reserve( String *str, uint32 size, bool relative = false, bool block = false ) const;
   virtual const Base *bufferedManipulator() const;
};

class FALCON_DYN_CLASS Buffer16: public Buffer
{
public:
   virtual uint32 charSize() const { return 2; }
   virtual uint32 length( const String *str ) const;
   virtual uint32 getCharAt( const String *str, uint32 pos ) const;
   virtual void setCharAt( String *str, uint32 pos, uint32 chr ) const;
   virtual void insert( String *str, uint32 pos, uint32 len, const String *source ) const;
};

class FALCON_DYN_CLASS Buffer32: public Buffer16
{
public:
   virtual uint32 charSize() const { return 4; }
   virtual uint32 length( const String *str ) const;
   virtual uint32 getCharAt( const String *str, uint32 pos ) const;
   virtual void setCharAt( String *str, uint32 pos, uint32 chr ) const;
   virtual void insert( String *str, uint32 pos, uint32 len, const String *source ) const;
};

extern FALCON_DYN_SYM Static handler_static;
extern FALCON_DYN_SYM Buffer handler_buffer;
extern FALCON_DYN_SYM Static16 handler_static16;
extern FALCON_DYN_SYM Buffer16 handler_buffer16;
extern FALCON_DYN_SYM Static32 handler_static32;
extern FALCON_DYN_SYM Buffer32 handler_buffer32;

} // namespace csh

/** Core string
   This class is called "Core String" because it represents the strings as the internal VM and engine
   sees them. This class is highly configurable and may manage any string that Falcon will ever need
   to mangle with.

   A set of fields is used to store the informations about the memory buffer where the string is
   actually held. The "kind" of string is determined by its manager. The manager is a special friend
   class that is in charge to effect all the needed operations on a particular kind of string. In
   example, there's a manager for static C strings, one for memAlloc() allocated strings, and in
   future also for chunked (multi buffer) stings and a parallel set of managers for international
   strings.

   The kind of the string can be changed by just changing its manager; this is often done automatically
   by an appropriate constructor or when some operation occour (i.e. a static string may be turned into
   a chunked one at write operations, and a chunked may get transformed into a buffered one if a linear
   access on the whole string is needed.

   String have a set of specialized subclasses which actually does nothing if not construct the
   base String with the appropriate string manager. Every corestring class is BOUND having not
   any private data member, because the derived String may be turned in something else at every moment
   without changing its memory position or layout. There's no RTTI information about this changes; all
   the polimorphism needed is applied by changing the string manager.

   However, String sublcass may define some new function members to handle initialization steps
   before "unmasking" the String structure and handle it back to the rest of the system. Also,
   as the String subclass may be determined by looking at the manager, a subclass with special
   operations (new member function) may be casted later on safely. The only requisite is that there's a 1:1
   mapping between corestring subclasses and the manager they use.
*/

class FALCON_DYN_CLASS String: public BaseAlloc //: public Garbageable
{

friend class csh::Base;
friend class csh::Byte;
friend class csh::Static;
friend class csh::Buffer;
friend class csh::Static16;
friend class csh::Buffer16;
friend class csh::Static32;
friend class csh::Buffer32;

friend class GarbageString;

protected:
   const csh::Base *m_class;
   uint32 m_allocated;
   uint32 m_size;
   uint32 m_id;
   byte *m_storage;

   /** True if this is a garbage string. */
   bool m_garbageable;

   /** gcStatus used by the GarbageString subclass.
      Placed here in the base class to help memory alignment.
   */
   byte m_gcStatus;

   /** Creates the core string.
      This method is protected. It can be accessed only by subclasses.
   */
   String( csh::Base *cl ):
      m_class( cl ),
      m_garbageable( false )
   {}

   /** Eventually fix the string allocation in the garbage collector.
      Mainly meant to be called by manipulators.
      \param oldSize the previous allocated size
   */
   void checkAdjustSize( uint32 oldSize );

public:

   enum constants {
      npos = csh::npos
   };

   /** Creates an empty string.
      The string is created non-zero terminated with length 0. It has also
      no valid internal storage at creation time.
   */
   String():
      m_size( 0 ),
      m_storage( 0 ),
      m_allocated( 0 ),
      m_class( &csh::handler_static ),
      m_garbageable( false )
   {}


   /** Adopt a static buffer as the internal buffer.
      This version of the string adopts the given buffer and becomes a "static string".

      A static string is just meant to carry around a pre-existing unchangeable (read-only)
      static buffer. The passed buffer must stay vaild for the whole duration of this
      string (i.e. it may be allocated as static string in some module).

      The string is automatically "bufferized" when some write operations are performed,
      so the original static data stays untouched even if this string is modified.

      This constructor allows for automatic fast char-to-string conversion in temporary
      operations.

      \note No assumption is made of the encoding of the source string. The data is
      just accepted as a mere sequence of bytes.

      \note The method bufferize() may be used later to force copy of the contents of this
            string. In that case, the underlying data must just stay valid until bufferize()
            is called.

      \see adopt

      \param data the source data to be copied
   */
   String( const char *data );

   /** Adopt a static buffer as the internal buffer.
      This is the wide char version.
      This version of the string adopts the given buffer and becomes a "static string".

      A static string is just meant to carry around a pre-existing unchangeable (read-only)
      static buffer. The passed buffer must stay vaild for the whole duration of this
      string (i.e. it may be allocated as static string in some module).

      The string is automatically "bufferized" when some write operations are performed,
      so the original static data stays untouched even if this string is modified.

      This constructor allows for automatic fast char-to-string conversion in temporary
      operations.

      \note No assumption is made of the encoding of the source string. The data is
         just accepted as a mere sequence of wide characters.

      \note The method bufferize() may be used later to force copy of the contents of this
            string. In that case, the underlying data must just stay valid until bufferize()
            is called.

      \see adopt

      \param data the source data to be copied
   */
   String( const wchar_t *data );


   /** Allows on-the-fly core string creation from static data.

      The resulting string is a bufferized copy of the static data; the source
      may be destroyed or become invalid, while this string will be still useable.

      \note To adopt an undestroyable buffer, use String( const char* ) version.
      \note No assumption is made of the encoding of the source string. The data is
      just accepted as a mere sequence of bytes.

      \param data the source data to be copied
      \param len the length of the string in buffer (in bytes). Pass -1 to make the constructor to determine the
            buffer length by scanning it in search for '\\0'
   */
   String( const char *data, int32 len );

   /** Allows on-the-fly core string creation from static data.
      This is the wide string version.

      The resulting string is a bufferized copy of the static data; the source
      may be destroyed or become invalid, while this string will be still useable.
      \note To adopt an undestroyable buffer, use String( const wchar_t* ) version.
      \note No assumption is made of the encoding of the source string. The data is
      just accepted as a mere sequence of wide characters.

      \param data the source data to be copied
      \param len the length of the buffer (in wide characters). Pass -1 to make the constructor to determine the
            buffer length by scanning it in search for '\\0'
   */
   String( const wchar_t *data, int32 len );


   /** Creates a bufferized string with preallocated space.
   */
   explicit String( uint32 prealloc );

   /** Copies a string.
      If the copied string is a bufferized string, a new bufferzied string is
      created, else a static string pointing to the same location of the original
      one is created.

      \note Static strings are constructed by simpling pointing the other string
      start position. Remember: static strings are meant to "carry" underlying
      memory and interpret it as a string, so the underlying memory must stay
      valid.

      Use bufferize() on this string to ensure that it is deep-copied.
   */
   String( const String &other ):
      m_garbageable( false ),
      m_allocated( 0 )
   {
      copy( other );
   }


   /** Substring constructor.
      This constructor is used to extract a substring from the original one,
      and is used in the subString() metod to return a string & as an inline call.
      Being an inline call, the optimized version optimizes the involved copy away.
      However, a string copy will still be present in debug.

      \note Static strings are constructed by simpling pointing the other string
      start position. Remember: static strings are meant to "carry" underlying
      memory and interpret it as a string, so the underlying memory must stay
      valid.

      Use bufferize() on this string to ensure that it is copied.
   */
   String( const String &other, uint32 begin, uint32 end = csh::npos );

   /** Destroys the String.
      As the method is not virtual (neither the class is), different kind of strings
      are destroyed by calling the destroy() method of their manipulators.
   */
   ~String()
   {
      m_class->destroy( this );
   }

   /** Copies the original string as-is.
      If the original string is of a static type, the buffer is just
      referenced, else a deep copy is performed.
      \param other the string to be copied.
      \return itself
   */
   void copy( const String &other );

   /** Creates a String forcing bufferization of the original one.
      This function copies the foreign string in a buffer responding to the
      toCstring requirements (zero terminate 8-bit strings, i.e. char* or UTF8).
      As clone and copy (and copy constructor) try to preserve remote string static
      allocation, this function is required when a bufferized copy is explicitly
      needed.
      \param other the string to be copied.
      \return itself
   */
   String &bufferize( const String &other );

   /** Forces this string to get buffer space to store even static strings.
      \return itself
   */
   String &bufferize();

    /** Adopt a pre-allocated dynamic buffer.
      This function takes the content of the given buffer and sets it as the
      internal storage of the string. The buffer is considered dynamically
      allocated with memAlloc(), and will be destroyed with memFree().

      This string is internally transformed in a raw buffer string; any
      previous content is destroyed.

      String is considered a single byte char width string.

      \param buffer the buffer to be adopted
      \param size the size of the string contained in the buffer (in bytes)
      \param allocated the size of the buffer as it was allocated (in bytes)
      \return itself
   */
   String &adopt( char *buffer, uint32 size, uint32 allocated );

   /** Adopt a pre-allocated dynamic buffer (wide char version).
      This function takes the content of the given buffer and sets it as the
      internal storage of the string. The buffer is considered dynamically
      allocated with memAlloc(), and will be destroyed with memFree().

      This string is internally transformed in a raw buffer string; any
      previous content is destroyed.

      String is considered a wide char width string.

      \param buffer the buffer to be adopted
      \param size the size of the string contained in the buffer (in character count)
      \param allocated the size of the buffer as it was allocated (in bytes)
      \return itself
   */
   String &adopt( wchar_t *buffer, uint32 size, uint32 allocated );


   /** Return the manipulator of the class.
      The manipulator is the function vector (under the form of a pure-virtual without-data class)
      that is used to handle the string. The manipulator identifies also the type of string,
      and so the subclass of the String that this strings currently belongs to.
   */
   const csh::Base *manipulator() const { return m_class; }

   /** Set the manipulator.
      Changing the manipulator also changes the meaning of the class deep data, and finally the
      correct subclass of String to which this item can be safely casted.
      Actually this method should be called only by internal functions, or only if you are
      really knowing what you are doing.
   */
   void manipulator( csh::Base *m ) { m_class = m; }

   /** Return the type of the string.
      The type is determined by the manipulator. Warning: this method calls a function virtually,
      so is quite slow. Better use it rarely. An acid test could be performed by matching the
      manipulator pointer with the standard manipulators.
   */
   csh::t_type type() const { return m_class->type(); }

   /** Returns the amount of allocated memory in the deep buffer.
      Used in buffers strings or in general in contiguous memory strings. Other kind of strings
      may ignore this (and so let it undefined) or use it for special purposes (i.e. amount of
      free memory on the last chunk in chunked strings.)
   */
   uint32 allocated() const { return m_allocated; }


   /** Changes the amount of allocated memory.
      Used in buffers strings or in general in contiguous memory strings. Other kind of strings
      may ignore this (and so let it undefined) or use it for special purposes (i.e. amount of
      free memory on the last chunk in chunked strings.)
   */
   void allocated( uint32 s ) { m_allocated = s; }

   /** Returns the amount of bytes the string occupies.
      This is the byte-size of the string, and may or may not be the same as the string length.
   */
   uint32 size() const { return m_size; }
   /** Changes the amount of bytes the string is considered to occupy.
      This is the byte-size of the string, and may or may not be the same as the string length.
   */
   void size( uint32 s ) { m_size = s; }

   /** Return the raw storage for this string.
      The raw storage is where the strings byte are stored. For more naive string (i.e. chunked), it
      may return the pointer to a structure helding more informations about the string data.
   */
   byte *getRawStorage() const { return m_storage; }

   /** Changes the raw storage in this string.
      This makes the string to point to a new memory position for its character data.
   */
   void setRawStorage( byte *b ) { m_storage = b; }

   /** Changes the raw storage in this string.
      This makes the string to point to a new memory position for its character data.
   */
   void setRawStorage( byte *b, int size ) {
      m_storage = b;
      m_size = size;
      m_allocated = size;
   }

   /** Return the length of the string in characters.
      The string length may vary depending on the string manipulator. This function calls
      a method in the manipuplator that selects the right way to calculate the string
      character count.
      Being not a pure accessor, is actually better to cache this value somewhere if repeteadly
      needed.
   */
   uint32 length() const { return m_class->length( this ); }

   /** Tranforms the string into a zero-terminated string.
      This function fills a buffer that can be fed in libc and STL function requiring a zero
      terminated string. The string manager will ensure that the data returned has one zero
      at the end of the string.

      8-bit strings are left unchanged.

      International strings are turned into UTF-8 strings (so that they can be fed
      into internationalized STL and libc functions).

      The operation is relatively slow. Use when no other option is avalaible, and cache
      the result.

      Provide a reasonable space. Safe space is size() * 4 + 1.

      \param target the buffer where to place the C string.
      \param bufsize the size of the target buffer in bytes.
      \return -1 if the buffer is not long enough, else returns the used size.
   */
   int32 toCString( char *target, uint32 bufsize ) const;

   /** Tranforms the string into a zero-terminated wide string.
      This function returns fills a buffer that can be fed in functions accpeting
      wchar_t strings. Returned strings are encoded in fixed lenght UTF-16, with
      endianity equivalent to native platform endianity.
      Character from extended planes are rendered as a single <?> 0x003f.

      The operation is relatively slow. Use when no other option is avalaible, and cache
      the result.

      Required space is constant, and exactly (lenght() + 1) * sizeof(wchar_t)  bytes
      (last "+1" is for final wchar_t "0" marker).

      \param target the buffer where to place the wchar_t string.
      \param bufsize the size of the target buffer in bytes.
      \return -1 if the buffer size is not large enough, else returns the string length in wchar_t count
   */
   int32 toWideString( wchar_t *target, uint32 bufsize ) const;

   /** Reduces the size of allocated memory to fit the string size.
      Use this method to shrink the allocated buffer storing the string
      to the minimal size needed to hold the string.
      This is useful when i.e. a buffered string was allocated to
      provide extra space for more efficient iterative appending,
      and the iterative appending is over.

      This has no effect on static string.

      \note use wisely as the shrink operation may require a string copy.
   */
   void shrink() { m_class->shrink( this ); }


   int32 id() const { return m_id; }
   void id( int32 val ) { m_id = val; }

   uint32 getCharAt( uint32 pos ) const { return m_class->getCharAt( this, pos ); }
   void setCharAt( uint32 pos, uint32 chr ) { m_class->setCharAt( this, pos, chr ); }
   String subString( int32 start, int32 end ) const { return String( *this, start, end ); }
   String subString( int32 start ) const { return String( *this, start, length() ); }
   bool change( int32 start, const String *other ) { return m_class->change( this, start, csh::npos, other ); }
   bool change( int32 start, int32 end, const String *other ) {
      return m_class->change( this, start, end, other );
   }
   void insert( uint32 pos, uint32 len, const String &source ) { m_class->insert( this, pos, len, &source ); }
   void remove( uint32 pos, uint32 len ) { m_class->remove( this, pos, len ); }
   void append( const String &source );
   void append( uint32 chr );
   void prepend( uint32 chr );

   void prepend( const String &source ) { m_class->insert( this, 0, 0, &source ); }
   String *clone() const { return m_class->clone( this ); }

   uint32 find( const String &element, uint32 start=0, uint32 end=csh::npos) const
   {
      return m_class->find( this, &element, start, end );
   }

   uint32 rfind( const String &element, uint32 start=0, uint32 end=csh::npos) const
   {
      return m_class->rfind( this, &element, start, end );
   }

   /** Compares a string to another.
      Optimized to match against C strings.
      \see compare( const String &other )
      \param other the other string to be compared
      \return -1 if this is less than the other, 0 if it's the same, 1 if it's greater.
   */
   int compare( const char *other ) const;

   /** Compares a string to another.
      Optimized to match against wide characters C strings.
      \see compare( const String &other )
      \param other the other string to be compared
      \return -1 if this is less than the other, 0 if it's the same, 1 if it's greater.
   */
   int compare( const wchar_t *other ) const ;

   /** Compares a string to another ignoring the case.
      This metod returns -1 if this string is less than the other,
      0 if it's the same and 1 if it's greater.

      \param other the other string to be compared
      \return -1 if this is less than the other, 0 if it's the same, 1 if it's greater.
   */
   int compare( const String &other ) const;

   /** Compares a string to another ignoring the case.
      This metod returns -1 if this string is less than the other,
      0 if it's the same and 1 if it's greater.

      Before checking them, uppercase characters are converted in
      the equivalent lowercase version; in this way "aBc" and "AbC"
      are considered the same.

      TODO - more caseization of accentuated letters

      \param other the other string to be compared
      \return -1 if this is less than the other, 0 if it's the same, 1 if it's greater.
   */
   int compareIgnoreCase( const String &other ) const;

   /** Compares a string to another ignoring the case.
      Optimized to match against C strings.
      \see compareIgnoreCase( const String &other )
      \param other the other string to be compared
      \return -1 if this is less than the other, 0 if it's the same, 1 if it's greater.
   */
   int compareIgnoreCase( const char *other ) const;

   /** Compares a string to another ignoring the case.
      Optimized to match against C strings.
      \see compareIgnoreCase( const String &other )
      \param other the other string to be compared
      \return -1 if this is less than the other, 0 if it's the same, 1 if it's greater.
   */
   int compareIgnoreCase( const wchar_t *other ) const;

   /** Returns true if this string is empty. */
   bool operator !() { return m_size == 0; }

   String & operator+=( const String &other ) { append( other ); return *this; }
   String & operator+=( uint32 other ) { append( other ); return *this; }
   String & operator+=( char other ) { append( (uint32) other ); return *this; }
   String & operator+=( const char *other ) { append( String( other ) ); return *this; }
   String & operator+=( wchar_t other ) { append( (uint32) other ); return *this; }
   String & operator+=( const wchar_t *other ) { append( String( other ) ); return *this; }

   String & operator=( const String &other ) {
      if ( m_allocated != 0 )
         m_class->destroy( this );
      copy( other );
      return *this;
   }

   /** Assign from a const char string.
      If this string is not empty, its content are destroyed; then
      this object is changed into a static zero terminated C string and
      the phisical location of the const char assigned to this string
      is taken as undestroyable reference. This operation is meant for
      C string phisically stored somewhere in the program and that stay
      valid for the whole program duration, or at least for the whole
      lifespan of this Falcon::String object.
   */
   String & operator=( const char *other ) {
      if ( m_storage != 0 )
         m_class->destroy( this );
      copy( String( other ) );
      return *this;
   }

   /** Order predicate.
      This predicate is used to sort Falcon::String objects and is provided
      mainly as an interface for the stl container classes.
      \param other the other string to check for
      \return true if this string is considered less (smaller in collation order)
             than the other one.
   */
   bool less( const String &other ) const { return compare( other ) < 0; }

   /** Save the string to a stream.
      The function never fails, but on failure something weird will happen
      on the stream. This may raise an exception, if the exceptions on the
      stream are turned on.
      \param out the stream on which to save the string.
   */
   void serialize( Stream *out ) const;

   /** Load the string from a stream.
      The string is deserialized from the stream and allocated in memory.
      This means that if the original was a static string, the deserialized
      one will be a string buffer of compatible type.

      If the string cannot be de-serialized the function returns false and
      the value is left as it were before calling the function. If the de
      serialization is succesful, the bufferized string is initializated
      and the function returns true.

      A failure usually means a stream corruption or an incompatible format.

      \param in the input stream where the string must be read from
      \return true on success, false on failure.
   */
   bool deserialize( Stream *in );

   /** Escapes a string for external representation.
      Convert special control characters to "\" prefixed characters,
      so tha the resulting string can be used in a source code to
      regenerate the same string in parsing.

      Characters below 0x0008 (backspace) are turned into hexadecimal
      representation, while international characters (from 0x0080 up)
      are left unchanged. This means that the resulting string must still
      be sent through an encoder to be safely saved on a stream.

      \param target the target string
   */
   void escape( String &target ) const;

   /** Escapes a string for external representation - full version.
      Convert special control characters to "\" prefixed characters,
      so tha the resulting string can be used in a source code to
      regenerate the same string in parsing.

      Characters below 0x0008 (backspace) and international characters
      (from 0x0080 up) are turned into hexadecimal
      representation. This means that the resulting string can be
      safely written on an output file without concerns for final
      encoding.

      \param target the target string
   */
   void escapeFull( String &target ) const;

   /** Unescape this string.
      Unescaping string is always an operation that leaves the string
      unchanged or shortened, so it can be done in place. Static
      strings are converted into buffered strings only if some
      actual unescape takes place.

      String unescaping understands special codes \\", \\\\, \\\\r, \\\\n, \\\\t and \\\\b,
      octal numbers \\0nnnn and hexadecimal numbers as \\xnnnnn, up to 32 bit
      precision.
   */
   void unescape();

   /** Unescape this string placing the result in another one.
      \see unescape()
   */
   void unescape( String &other ) const
   {
      other = *this;
      other.unescape();
   }

   /** Minimal numerical conversion.
      If this string represents a valid integer, the integer is returned.
      The string is considered a valid integer also if it is followed by non-numerical data.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseInt( int64 &target, uint32 pos = 0 ) const;

   /** Minimal numerical conversion.
      If this string represents a valid integer in octal format, the integer is returned.
      Pos must start after the octal marker \\0.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseOctal( uint64 &target, uint32 pos = 0 ) const;

   /** Minimal numerical conversion.
      If this string represents a valid integer in hexadecimal format, the integer is returned.
      Pos must start after the octal marker \\x.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseHex( uint64 &target, uint32 pos = 0 ) const;

   /** Minimal numerical conversion.
      If this string represents a valid floating point number, the number is returned.
      The string is considered a valid number also if it is followed by non-numerical data.
      Floating point number may be in scientific notation.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseDouble( double &target, uint32 pos = 0 ) const;


    /** Converts a number to a string and appends it to this string.
      \param number the number to be converted.
    */
    void writeNumber( int64 number );

    /** Converts a number to a string and appends it to this string.
      This version writes the number in hex format.
      \param number the number to be converted.
      \param uppercase true to use ABCDEF letters instead of abdef
    */
    void writeNumberHex( uint64 number, bool uppercase = true );

    /** Converts a number to a string and appends it to this string.
      This version writes the number in octal format.
      \param number the number to be converted.
    */
    void writeNumberOctal( uint64 number  );

    /** Converts a number to a string and appends it to this string.
      Number is formatted with prinf format "%e"; to specify a different
      format, use the other version of this method.
      \param number the number to be converted.
    */
    void writeNumber( double number )
    {
      writeNumber( number, "%e" );
    }

    /** Converts a number to a string and appends it to this string.
      This version allows to specify a format to be passed to sprintf
      in the conversion of the number.
      Regardless of the fact that sprintf writes a string int 8-bit
      char space, both the format string and this string where the
      result is to be appended may be in any char width.

      The function does not check for validity of the format.

      \param number the number to be converted.
      \param format the format to be passed to printf for conversion.
    */
    void writeNumber( double number, const String &format );

    /** Checks the position to be in the string, and eventually changes it if it's negative.
      This is just a nice inline shortuct so that the string constructor for substrings
      can be called safely.
      \param pos the position to be checked and eventually turned into a positive value.
      \return false if pos is outside thte string
   */
    bool checkPosBound( int32 &pos )
    {
      register int s = length();
      if ( pos < 0 )
         pos = s + pos;
      if ( pos < 0 || pos >= s )
         return false;
      return true;
    }

    /** Checks the range to be in the string, and eventually changes it if it's negative.
      This is just a nice inline shortuct so that the string constructor for substrings
      can be called safely.
      \param begin the start to be checked and eventually turned into a positive value.
      \param end the end to be checked and eventually turned into a positive value.
      \return false if the range cannot be mapped in string.
   */
    bool checkRangeBound( int32 &begin, int32 &end )
    {
      register int s = length();
      if ( begin < 0 )
         begin = s + begin;
      if ( begin < 0 || begin >= s )
         return false;
      if ( end < 0 )
         end = s + end + 1;

      // end can be the same as lenght
      if ( end < 0 || end > s )
         return false;
      return true;
    }

    /** Reserve buffer space in the target string.
      This ensures that the space allocated in the string is bufferzed and at least
      wide enough to store the requested bytes.
      \note the width is in bytes.
      \param size minimal space that must be allocated as writeable heap buffer (in bytes).
   */
   void reserve( uint32 size )
   {
      m_class->reserve( this, size );
   }

   /** Check if this instance of string is actually an instance of GarbageString.
      \return true if the instance is derived from GarbageString and can be statically casted to that class.
   */
   bool garbageable() const { return m_garbageable; }

   /** Remove efficiently whitespaces at beginning and end of the string.
      If whitespaces are only at the end of the string, the lenght of the string
      is simply reduced; this means that static strings may stay static after
      this process.
      In case of whitespaces at beginning, the string will be resized, and eventually
      allocated, moving the characters back to the beginning position.

   */
   void trim();

   bool isStatic() const {
      return manipulator()->type() == csh::cs_static ||
             manipulator()->type() == csh::cs_static16 ||
             manipulator()->type() == csh::cs_static32;
   }

   /** Bufferize an UTF-8 string.

      This is an efficient shortcut for the very common case of UTF8 strings
      being turned into falcon string.
      There isn't a drect constructor that understands that the input char *
      is an UTF8 string, but the proxy generators UTF8String and UTF8GarbageString
      serve to this purpose.

      After the call, the previous content of this string is destroyed.

      In case of an invalid UTF8 sequence, up to what is possible to decode is
      read, and the function return false.

      \param utf8 the utf8 string to be loaded
      \return true on success, false if the sequence is invalid.
   */
   bool fromUTF8( const char *utf8 );
};


/** Equality operator */
inline bool operator == ( const String &str1, const String &str2 )  { return str1.compare( str2 ) == 0; }
inline bool operator == ( const String &str1, const char *str2 )    { return str1.compare( str2 ) == 0; }
inline bool operator == ( const String &str1, const wchar_t *str2 ) { return str1.compare( str2 ) == 0; }
inline bool operator != ( const String &str1, const String &str2 )  { return str1.compare( str2 ) != 0; }
inline bool operator != ( const String &str1, const char *str2 )    { return str1.compare( str2 ) != 0; }
inline bool operator != ( const String &str1, const wchar_t *str2 ) { return str1.compare( str2 ) != 0; }
inline bool operator >  ( const String &str1, const String &str2 )  { return str1.compare( str2 ) > 0; }
inline bool operator >  ( const String &str1, const wchar_t *str2 ) { return str1.compare( str2 ) > 0; }
inline bool operator >  ( const String &str1, const char *str2 )    { return str1.compare( str2 ) > 0; }
inline bool operator <  ( const String &str1, const String &str2 )  { return str1.compare( str2 ) < 0; }
inline bool operator <  ( const String &str1, const char *str2 )    { return str1.compare( str2 ) < 0; }
inline bool operator <  ( const String &str1, const wchar_t *str2 ) { return str1.compare( str2 ) < 0; }
inline bool operator >= ( const String &str1, const String &str2 )  { return str1.compare( str2 ) >= 0; }
inline bool operator >= ( const String &str1, const char *str2 )    { return str1.compare( str2 ) >= 0; }
inline bool operator >= ( const String &str1, const wchar_t *str2 ) { return str1.compare( str2 ) >= 0; }
inline bool operator <= ( const String &str1, const String &str2 )  { return str1.compare( str2 ) <= 0; }
inline bool operator <= ( const String &str1, const char *str2 )    { return str1.compare( str2 ) <= 0; }
inline bool operator <= ( const String &str1, const wchar_t *str2 ) { return str1.compare( str2 ) <= 0; }

inline String operator +( const String &str1, const String &str2 )
   { String str3; str3.append( str1 ); str3.append( str2); return str3; }
inline String operator +( const char *str1, const String &str2 )
   { String str3; str3.append( str1 ); str3.append( str2); return str3; }
inline String operator +( const wchar_t *str1, const String &str2 )
   { String str3; str3.append( str1 ); str3.append( str2); return str3; }
inline String operator +( const String &str1, const char *str2 )
   { String str3; str3.append( str1 ); str3.append( str2); return str3; }
inline String operator +( const String &str1, const wchar_t *str2 )
   { String str3; str3.append( str1 ); str3.append( str2); return str3; }

/** Core string comparer class */
class StringPtrCmp
{
public:
   bool operator() ( const String *s1, const String *s2 ) const
      { return s1->compare( *s2 ) < 0; }
};


/** Garbageable string class.
   As Falcon::String is the most common garbageable item found in Falcon, and as it's
   also used widely in the engine and in the embedding applications without
   an express need for garbage collecting, this specialized garbage class
   has been provided expecially for strings.

   This forces the garbage collector to have two chains of garbageable items,
   the normal items and the strings, but as a great deal of strings in a falcon based
   applications are not menant to be garbage collected, the differentiation between
   garbageable strings and program strings pays back the extra complexity of this
   management.

   Also, having non garbageable Falcon::String as module and program application string class
   saves 16 bytes per string and obviates the need for the Garbageable virtual
   destructor, with another far ptr space spare for the virtual table, which are a
   relatively considerable amount of memory if compared to the size of strings
   usually managed in scripts.

   Finally, as strings has already an accounting system for their allocated memory,
   a specialized GarbageString which knows the underlying strings mechanics is more
   efficient than a generic Garbageable object accounting.
*/

class FALCON_DYN_CLASS GarbageString: public String
{
   GarbageString *m_garbage_next;
   GarbageString *m_garbage_prev;
   VMachine *m_origin;

public:
   /** Empty constructor.
      Creates an empty garbage string.

      \see String()
   */
   GarbageString( VMachine *vm );

   /** Preallocator constructor
      Creates an empty garbage string with preallocation support.

      \see String( uint32 )
   */
   explicit GarbageString( VMachine *vm, uint32 prealloc );


   /** Substring constructor
      Creates a garbage string which copies a part of the given string.

      The constructor is explicit: you can't use a char* or wchar_t* with automatic conversion
      into String as the \b other parameter of this constructor; it must be an already existing
      string or garbage string.

      \see String( const String &, uint32, uint32 )
   */
   explicit GarbageString( VMachine *vm, const String &other, uint32 pos0, uint32 pos1 = npos );

   /** Constructs a copy of the string
      Creates a garbage string copying the contents of the other string.
      If the source string is static, this GarbageString becomes a static
      string too. If it is a bufferized string, this string becomes bufferized too,
      and the contents of the other string are copied in the newly allocated buffer.
      \see String
   */
   explicit GarbageString( VMachine *vm,  const String &other );

    /** Adopt a static buffer and creates a static string.
      This constructor just makes this string to point to the static data that has been
      provided, which must stay valid for the whole lifetime of this string.
      \see String( const char* )
   */
   GarbageString( VMachine *vm, const char *other );

    /** Adopt a static buffer and creates a static string.

      This constructor just makes this string to point to the static data that has been
      provided, which must stay valid for the whole lifetime of this string.
      This is the wide char version.
      \see String( const wchar_t* )
   */
   GarbageString( VMachine *vm, const wchar_t *other );

   /** Copies the static data in this string.
      \see String( const char*, int32 )
   */
   GarbageString( VMachine *vm, const char *other, int32 len );

   /**  Copies the static data in this string.
      \see String( const wchar_t*, int32 )
   */
   GarbageString( VMachine *vm, const wchar_t *other, int32 len );

   /** Copy constructor.

   */
   GarbageString( const GarbageString &other );

   ~GarbageString() {}

   /** Mark for Garbage Collecting.
      This is reserved to the Garbage collector, don't use it.
   */
   void mark( byte mode ) {
      m_gcStatus = mode;
   }

   /** Return the current GC mark status. */
   unsigned char mark() {
      return m_gcStatus;
   }

   GarbageString *nextGarbage() const { return m_garbage_next; }
   GarbageString *prevGarbage() const { return m_garbage_prev; }
   void nextGarbage( GarbageString *next ) { m_garbage_next = next; }
   void prevGarbage( GarbageString *prev ) { m_garbage_prev = prev; }

   void updateGCSize( uint32 oldSize = 0 ) const;
};


//=============================
// inline forwarded decls
//

inline void String::checkAdjustSize( uint32 oldSize )
{
   if ( garbageable() && oldSize != m_allocated )
      static_cast<GarbageString *>( this )->updateGCSize( oldSize );
}

//=================================
// Helpful list of string deletor
//
void string_deletor( void *data );

//=================================
// inline proxy constructors
//

/** Creates a String from an utf8 sequence on the fly.
   This is a proxy constructor for String. The string data is
   bufferized, so the sequence needs not to be valid after this call.

   \note this function returns a valid string also if the \b utf8 paramter is
      not a valid utf8 sequence. If there is the need for error detection,
      use String::fromUTF8 instead.

   \see String::fromUTF8
   \param utf8 the sequence
   \return a string containing the decoded sequence
*/
inline String *UTF8String( const char *utf8 )
{
   String *str = new String;
   str->fromUTF8( utf8 );
   return str;
}

/** Creates a Garbage String from an utf8 sequence on the fly.
   This is a proxy constructor for GarbageString. The string data is
   bufferized, so the sequence needs not to be valid after this call.

   \note this function returns a valid string also if the \b utf8 paramter is
      not a valid utf8 sequence. If there is the need for error detection,
      use String::fromUTF8 instead.

   \see String::fromUTF8
   \param vm a VM where this garbage string will be stored
   \param utf8 the sequence
   \return a string containing the decoded sequence
*/

inline GarbageString *UTF8GarbageString( VMachine *vm, const char *utf8 )
{
   GarbageString *str = new GarbageString( vm );
   str->fromUTF8( utf8 );
   return str;
}

}

#endif

/* end of string.h */
