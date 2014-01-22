/*
   FALCON - The Falcon Programming Language.
   FILE: string.h

   Core falcon string representation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven nov 19 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core falcon string representation
*/

#ifndef _FALCON_STRING_H_
#define _FALCON_STRING_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <stdlib.h>

#define FALCON_STRING_ALLOCATION_BLOCK 32
#define FALCON_DEFAULT_NUMERIC_STRING_FORMAT "%.14g"

namespace Falcon {

class DataWriter;
class DataReader;
class GCToken;
class Class;

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
   cs_membuf
} t_type;

/** Invalid position for core strings. */
const length_t npos = -1;

/** Base corestring manager class.
   This is actually an interface that must be implemented by all the core string managers.
*/
class FALCON_DYN_CLASS Base
{
public:
   virtual ~Base() {}
   t_type type() const { return m_type; }
   uint32 charSize() const { return m_charSize; }   
   inline length_t length( const String *str ) const;
   
   virtual void setCharAt( String *str, length_t pos, char_t chr ) const =0;
   virtual void subString( const String *str, int32 start, int32 end, String *target ) const =0;
   /** Finds a substring in a string, and eventually returns npos if not found. */
   virtual void insert( String *str, length_t pos, length_t len, const String *source ) const =0;
   virtual bool change( String *str, length_t start, length_t end, const String *source ) const =0;
   virtual void remove( String *str, length_t pos, length_t len ) const =0;
   virtual String *clone( const String *str ) const =0;
   virtual void destroy( String *str ) const =0;

   virtual void bufferize( String *str ) const =0;
   virtual void bufferize( String *str, const String *strOrig ) const =0;
   virtual void toMemBuf( String *str ) const = 0;
   virtual void toMemBuf( String *str, const String *strOrig ) const = 0;
   virtual void shrink( String *str ) const = 0;

   virtual const Base *bufferedManipulator() const =0;
   virtual const Base *membufManipulator() const =0;

protected:
   t_type m_type;
   uint32 m_charSize;
};

/** Byte orientet base class.
   This is still an abstract class, but it provides minimal behavior for byte oriented
   strings (ascii or system specific).
*/
class FALCON_DYN_CLASS Byte: public Base
{
public:
   virtual ~ Byte() {}
   
   // Todo: fix this incongruence
   virtual void subString( const String *str, int32 start, int32 end, String *target ) const;
   virtual bool change( String *str, length_t pos, length_t end, const String *source ) const;
   virtual String *clone( const String *str ) const;
   virtual void remove( String *str, length_t pos, length_t len ) const;

   virtual void bufferize( String *str ) const;
   virtual void bufferize( String *str, const String *strOrig ) const;

   virtual void toMemBuf( String *str ) const;
   virtual void toMemBuf( String *str, const String *strOrig ) const;

   virtual const Base *bufferedManipulator() const { return this; }
   virtual const Base *membufManipulator() const { return this; }
};


/** Static byte oriented string manager.
   Useful to instantiate and manage strings whose content is byte oriented and whose size is
   known in advance; for example, symbol names in the Falcon module are easily managed with this class.

   Every write operation on strings managed by this class will cause its manager to be changed
   into the Buffer class.
*/
class FALCON_DYN_CLASS Static: public Byte
{
public:
   Static()
   {
      m_type = cs_static;
      m_charSize = 1;
   }
   
   virtual ~Static() {}

   virtual void setCharAt( String *str, length_t pos, char_t chr ) const;
   virtual void insert( String *str, length_t pos, length_t len, const String *source ) const;
   virtual void remove( String *str, length_t pos, length_t len ) const;
   virtual void destroy( String *str ) const;

   virtual void shrink( String *str ) const;
   virtual const Base *bufferedManipulator() const;
   virtual const Base *membufManipulator() const;
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
   Buffer()
   {
      m_type = cs_buffer;
      m_charSize = 1;
   }
   
   virtual ~Buffer() {}

   virtual void setCharAt( String *str, length_t pos, char_t chr ) const;
   virtual void insert( String *str, length_t pos, length_t len, const String *source ) const;
   virtual void destroy( String *str ) const;
   virtual void shrink( String *str ) const;
   virtual const Base *membufManipulator() const;
};

class FALCON_DYN_CLASS Static16: public Static
{
public:
   Static16()
   {
      m_charSize = 2;
   }
   
   virtual ~Static16() {}
   virtual void setCharAt( String *str, length_t pos, char_t chr ) const;
   virtual void remove( String *str, length_t pos, length_t len ) const;
   virtual const Base *bufferedManipulator() const;
   virtual const Base *membufManipulator() const;
};

class FALCON_DYN_CLASS Static32: public Static16
{
public:
   Static32()
   {
      m_charSize = 4;
   }

   virtual ~Static32() {}
   virtual void setCharAt( String *str, length_t pos, char_t chr ) const;
   virtual void remove( String *str, length_t pos, length_t len ) const;
   virtual const Base *bufferedManipulator() const;
   virtual const Base *membufManipulator() const;
};

class FALCON_DYN_CLASS Buffer16: public Buffer
{
public:
   Buffer16()
   {
      m_charSize = 2;
   }
   
   virtual ~Buffer16() {}
   
   virtual void setCharAt( String *str, length_t pos, char_t chr ) const;
   virtual const Base *membufManipulator() const;
};

class FALCON_DYN_CLASS Buffer32: public Buffer16
{
public:
   Buffer32()
   {
      m_charSize = 4;
   }
   virtual ~Buffer32() {}
   
   virtual void setCharAt( String *str, length_t pos, char_t chr ) const;
   virtual const Base *membufManipulator() const;
};


class FALCON_DYN_CLASS MemBuf: public Buffer
{
public:
   MemBuf()
   {
      m_type = cs_membuf;
      m_charSize = 1;
   }

   virtual ~MemBuf() {}

   virtual void setCharAt( String *str, length_t pos, char_t chr ) const;
   virtual void insert( String *str, length_t pos, length_t len, const String *source ) const;
   virtual const Base *bufferedManipulator() const;
};


class FALCON_DYN_CLASS MemBuf16: public MemBuf
{
public:
   MemBuf16():MemBuf(){}
   virtual ~MemBuf16() {}
   virtual const Base *bufferedManipulator() const;
};

class FALCON_DYN_CLASS MemBuf32: public MemBuf
{
public:
   MemBuf32():MemBuf(){}
   virtual ~MemBuf32() {}
   virtual const Base *bufferedManipulator() const;
};


extern FALCON_DYN_SYM Static handler_static;
extern FALCON_DYN_SYM Buffer handler_buffer;
extern FALCON_DYN_SYM Static16 handler_static16;
extern FALCON_DYN_SYM Buffer16 handler_buffer16;
extern FALCON_DYN_SYM Static32 handler_static32;
extern FALCON_DYN_SYM Buffer32 handler_buffer32;

extern FALCON_DYN_SYM MemBuf handler_membuf;
extern FALCON_DYN_SYM MemBuf16 handler_membuf16;
extern FALCON_DYN_SYM MemBuf32 handler_membuf32;

} // namespace csh



class FALCON_DYN_CLASS String
{

   friend class csh::Base;
   friend class csh::Byte;
   friend class csh::Static;
   friend class csh::Buffer;
   friend class csh::Static16;
   friend class csh::Buffer16;
   friend class csh::Static32;
   friend class csh::Buffer32;

protected:
   const csh::Base *m_class;
   length_t m_allocated;
   length_t m_size;
   byte *m_storage;
   uint32 m_lastMark;
   bool m_bImmutable;

   /**
    * Creates the string.
    *
    * This method is protected. It can be accessed only by subclasses.
    */
   explicit String( csh::Base *cl ) :
      m_class( cl ),
      m_bImmutable(false)
   {}

   void internal_escape( String &strout, bool full ) const;

public:

   enum constants {
      npos = csh::npos
   };


   /**
    * Creates an empty string.
    *
    * The string is created non-zero terminated with length 0. It has also
    * no valid internal storage at creation time.
    */
   String():
      m_class( &csh::handler_static ),
      m_allocated( 0 ),
      m_size( 0 ),
      m_storage( 0 ),
      m_lastMark( 0 ),
      m_bImmutable(false)
   {
   }


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
   String( const char *data, length_t len );

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
   String( const wchar_t *data, length_t len );


   /** Creates a bufferized string with preallocated space.
   */
   explicit String( length_t prealloc );

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
      m_allocated( 0 ),
      m_lastMark( other.m_lastMark ),
      m_bImmutable(false)
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
    // TODO fix size and reversing
   */
   String( const String &other, length_t begin, length_t end = csh::npos );

   /** Destroys the String.
      As the method is not virtual (neither the class is), different kind of strings
      are destroyed by calling the destroy() method of their manipulators.
   */
   ~String()
   {
      m_class->destroy( this );
   }

   static Class* handler();

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

   /** Copies the other string in a buffer.
      \return itself
   */
   String &toMemBuf( const String& other );

   /** Forces this string to get buffer space AND sets it to non-textual content.
      \return itself
   */
   String &toMemBuf();

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
   String &adopt( char *buffer, length_t size, length_t allocated );

   /** Adopt a pre-allocated dynamic buffer as non-textual content.
     This function takes the content of the given buffer and sets it as the
     internal storage of the string. The buffer is considered dynamically
     allocated with malloc(), and will be destroyed with free().

     This string is internally transformed in a raw buffer of non-text data;
     any previous content is destroyed.

     \param buffer the buffer to be adopted
     \param size the size of the string contained in the buffer (in bytes)
     \param allocated the size of the buffer as it was allocated (in bytes)
     \return itself
  */
   String &adoptMemBuf( byte *buffer, length_t size, length_t allocated );

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
   String &adopt( wchar_t *buffer, length_t size, length_t allocated );


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
   void manipulator( const csh::Base *m ) { m_class = m; }

   /** Return the type of the string.
      The type is determined by the manipulator. Warning: this method calls a function virtually,
      so is quite slow. Better use it rarely. An acid test could be performed by matching the
      manipulator pointer with the standard manipulators.
   */
   csh::t_type type() const { return m_class->type(); }

   /**
       Returns true if the string is meant to hold text.

       This returns true if the string was created as holding
       textual data (either static or buffered).

       \return true if the content is textual, false if it's an arbitrary
       binary data memory buffer.
    */
   bool isText() const { return m_class->type() != csh::cs_membuf; }
   bool isMemBuf() const { return m_class->type() == csh::cs_membuf; }

   /**
       Return true if the string is static.

       \return true if the string is using a static buffer.
    */
   bool isStatic() const { return m_class->type() == csh::cs_static; }

   /** Returns the amount of allocated memory in the deep buffer.
      Used in buffers strings or in general in contiguous memory strings. Other kind of strings
      may ignore this (and so let it undefined) or use it for special purposes (i.e. amount of
      free memory on the last chunk in chunked strings.)
   */
   length_t allocated() const { return m_allocated; }


   /** Changes the amount of allocated memory.
      Used in buffers strings or in general in contiguous memory strings. Other kind of strings
      may ignore this (and so let it undefined) or use it for special purposes (i.e. amount of
      free memory on the last chunk in chunked strings.)
   */
   void allocated( length_t s ) { m_allocated = s; }

   /** Returns the amount of bytes the string occupies.
      This is the byte-size of the string, and may or may not be the same as the string length.
   */
   length_t size() const { return m_size; }
   /** Changes the amount of bytes the string is considered to occupy.
      This is the byte-size of the string, and may or may not be the same as the string length.
   */
   void size( length_t s ) { m_size = s; }

   /** Resets the string to 0 size and charlenght 1, without destroying the allocated memory */
   void clear();

   /** Efficient check for string emptyness.
    * \return true if the string is empty.
    */
   bool empty() const { return m_size == 0; }

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
   length_t length() const { return m_class->length( this ); }

   /** Transforms the string into a zero-terminated string.
      This function fills a buffer that can be fed in libc and STL function
      requiring a zero terminated string.
      The string manager will ensure that the data returned has one zero
      at the end of the string.

      International strings are turned into UTF-8 strings (so that they can be fed
      into internationalized STL and libc functions).

      The operation is relatively slow. Use when no other option is available,
      and cache the result.

      Provide a reasonable space. Safe space is size() * 4 + 1.

      \note Consider using AutoCString helper class for automated allocation
      in stack or in heap of the needed memory.

      \param target the buffer where to place the C string.
      \param bufsize the size of the target buffer in bytes.
      \return npos if the buffer is not long enough, else returns the used size.
   */
   length_t toUTF8String( char *target, length_t bufsize ) const
   {
      return toCString( target, bufsize );
   }

   /**
     Alias to toUTF8String.

      \param target the buffer where to place the C string.
      \param bufsize the size of the target buffer in bytes.
      \return npos if the buffer is not long enough, else returns the used size.
    */
   length_t toCString( char *target, length_t bufsize ) const;

   /** Transforms the string into a zero-terminated wide string.
      This function returns fills a buffer that can be fed in functions accpeting
      wchar_t strings. Returned strings are encoded in fixed lenght UTF-16, with
      endianity equivalent to native platform endianity.
      Character from extended planes are rendered as a single <?> 0x003f.

      The operation is relatively slow. Use when no other option is avalaible, and cache
      the result.

      Required space is constant, and exactly (lenght() + 1) * sizeof(wchar_t)  bytes
      (last "+1" is for final wchar_t "0" marker).

      \note Consider using AutoWString helper class for automated allocation
      in stack or in heap of the needed memory.

      \param target the buffer where to place the wchar_t string.
      \param bufsize the size of the target buffer in bytes.
      \return npos if the buffer size is not large enough, else returns the string length in wchar_t count
   */
   length_t toWideString( wchar_t *target, length_t bufsize ) const;

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

   inline char_t getCharAt( length_t pos ) const
   {
      switch( m_class->charSize() )
      {
         case 1: return getRawStorage()[pos];
         case 2: return ((uint16*)getRawStorage())[pos];
         case 4: return ((uint32*)getRawStorage())[pos];
      }
      return 0;
   }
   
   void setCharAt( length_t pos, char_t chr ) { m_class->setCharAt( this, pos, chr ); }
   String subString( int32 start, int32 end ) const { return String( *this, start, end ); }
   String subString( int32 start ) const { return String( *this, start, length() ); }
   bool change( length_t start, const String &other ) {
      return m_class->change( this, start, csh::npos, &other );
   }
   bool change( length_t start, length_t end, const String &other ) {
      return m_class->change( this, start, end, &other );
   }
   void insert( length_t pos, length_t len, const String &source ) { m_class->insert( this, pos, len, &source ); }
   void remove( length_t pos, length_t len ) { m_class->remove( this, pos, len ); }
   void append( const String &source );
   void append( char_t chr );
   void prepend( char_t chr );

   void prepend( const String &source ) { m_class->insert( this, 0, 0, &source ); }

   length_t find( const String &element, length_t start=0, length_t end=csh::npos) const;
   length_t rfind( const String &element, length_t start=0, length_t end=csh::npos) const;
   length_t find( char_t element, length_t start=0, length_t end=csh::npos) const;
   length_t rfind( char_t element, length_t start=0, length_t end=csh::npos) const;

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

   /** Find one of the characters in the string. 
   \param src A string containing all the characters to be searched.
    */
   length_t findFirstOf( const String& src, length_t pos = 0 ) const;

   /** Find one of the characters in the string from the back of the string. 
    \param src A string containing all the characters to be searched.
    */
   length_t findLastOf( const String& src, length_t pos = npos ) const ;
   
   /** Returns true if this string is empty. */
   bool operator !() { return m_size == 0; }

   String & operator+=( const String &other ) { append( other ); return *this; }
   String & operator+=( char_t other ) { append( other ); return *this; }
   String & operator+=( char other ) { append( (char_t) other ); return *this; }
   String & operator+=( const char *other ) { append( String( other ) ); return *this; }
   String & operator+=( wchar_t other ) { append( (char_t) other ); return *this; }
   String & operator+=( const wchar_t *other ) { append( String( other ) ); return *this; }

   String & operator=( const String &other ) {
      copy( other );
      return *this;
   }

   String & operator=( char_t chr ) {
      m_size = 0;
      append( chr );
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
   void serialize( DataWriter *out ) const;

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
   bool deserialize( DataReader *in );

   /** Escapes a string for external representation.
      Convert special control characters to "\" prefixed characters,
      so that the resulting string can be used in a source code to
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
      \param target The target string where to perform the unescape.

      String unescaping understands special codes \\", \\\\, \\\\r, \\\\n, \\\\t and \\\\b,
      octal numbers \\0nnnn and hexadecimal numbers as \\xnnnnn, up to 32 bit
      precision.
   */
   void unescape(String& target);

   /** Minimal numerical conversion.
      If this string represents a valid integer, the integer is returned.
      The string is considered a valid integer also if it is followed by non-numerical data.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseInt( int64 &target, length_t pos = 0 ) const;

   /** Minimal numerical conversion.
      If this string represents a valid integer in octal format, the integer is returned.
      Pos must start after the octal marker \\0 or \\c.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseOctal( uint64 &target, length_t pos = 0 ) const;

   /** Minimal numerical conversion.
      If this string represents a valid integer in octal format, the integer is returned.
      Pos must start after the octal marker \\b.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseBin( uint64 &target, length_t pos = 0 ) const;

   /** Minimal numerical conversion.
      If this string represents a valid integer in hexadecimal format, the integer is returned.
      Pos must start after the octal marker \\x.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseHex( uint64 &target, length_t pos = 0 ) const;

   /** Minimal numerical conversion.
      If this string represents a valid floating point number, the number is returned.
      The string is considered a valid number also if it is followed by non-numerical data.
      Floating point number may be in scientific notation.
      \param target place where to store the number
      \param pos initial position in the string from which to start the conversion
      \return true if succesful, false if parse failed
   */
   bool parseDouble( double &target, length_t pos = 0 ) const;


    /** Converts a number to a string and appends it to this string.
      \param number the number to be converted.
    */
    void writeNumber( int64 number );

    /** Converts a number to a string and appends it to this string.
      This version writes the number in hex format.
      \param number the number to be converted.
      \param uppercase true to use ABCDEF letters instead of abdef
    */
    void writeNumberHex( uint64 number, bool uppercase = true, int count = 0 );

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
      writeNumber( number, FALCON_DEFAULT_NUMERIC_STRING_FORMAT );
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

    void writeNumber( int64 number, const String &format );

    /** Cumulative version of writeNumber.
     *
     * This method can be used to concatenate strings and number such as for
     * String s = String( "You got ").N( msg_count ).A( " messages " );
     */
    inline String& N( int64 number )
    {
       writeNumber( number );
       return *this;
    }

    /** Cumulative version of writeNumber.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */
    inline String& N( int32 number )
    {
       writeNumber( (int64) number );
       return *this;
    }

    /** Cumulative version of writeNumber.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */
    inline String& N( uint32 number )
    {
       writeNumber( (int64) number );
       return *this;
    }

    /** Cumulative version of writeNumber.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */
    inline String& N( int64 number, const String& format )
    {
       writeNumber( (int64) number, format );
       return *this;
    }

    /** Cumulative version of writeNumber.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */
    inline String& N( double number )
    {
       writeNumber( number );
       return *this;
    }

    /** Cumulative version of writeNumber.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */
    inline String& N( double number, const String& format )
    {
       writeNumber( number, format );
       return *this;
    }

    /** Cumulative version of writeHex */
    inline String& H( uint64 number, bool ucase, int digits = 0 )
    {
       writeNumberHex( number, ucase, digits );
       return *this;
    }

    /** Cumulative version of append.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */

    inline String& A( const String& str ) { append(str); return *this; }

    /** Cumulative version of append.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */
    inline String& A( int chr )  { append((char_t)chr); return *this; }

    /** Cumulative version of append.
      *
      * This method can be used to concatenate strings and number such as for
      * String s = String( "You got ").N( msg_count ).A( " messages " );
      */
    inline String& A( char_t chr ) { append(chr); return *this; }

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
         end = s + end;

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
   void reserve( length_t size );
   
   typedef enum  {
      e_tm_all,
      e_tm_front,
      e_tm_back
   }
   t_trimmode;

   /** Remove efficiently whitespaces at beginning and end of the string.
      If whitespaces are only at the end of the string, the lenght of the string
      is simply reduced; this means that static strings may stay static after
      this process.
      In case of whitespaces at beginning, the string will be resized, and eventually
      allocated, moving the characters back to the beginning position.
      \param mode 0 = all, 1 = front, 2 = back

   */
   void trim( t_trimmode mode );

   /** Remove efficiently characters from a given set at beginning and end of the string.
      \param mode 0 = all, 1 = front, 2 = back
      \param set A string containing the characters to be removed.
   */
   void trimFromSet( t_trimmode mode, const String& set );
   
   /** Trims whitespaces from all the parts of the string. */
   void trim() { trim( e_tm_all ); }

   /** Returns a string replicating the first one. 
    \param times Count of replicates of this string
    \return a string conatining this string replicated n times.
    
    If times is 0 the resulting string is empty; if it's 1, its a copy of the
    original one.
    */
   String replicate( int times );
   
   /**
    * Remove efficiently 'what' at the beginning of the string.
    *
    * If what is empty, whitespaces are removed.
    */
   void frontTrim() { trim( e_tm_front ); }

   /**
    * Remove efficiently 'what' at the end of the string.
    *
    * If what is empty, whitespaces are removed.
    */
   void backTrim() { trim( e_tm_back ); }

   void frontTrimFromSet( const String& set )
   {
      trimFromSet( e_tm_front, set );
   }

   void backTrimFromSet( const String& set )
   {
      trimFromSet( e_tm_back, set );
   }

   /**
    * Convert the string to all lower case.
    */
   void lower();

   /**
    * Convert the string to all upper case.
    */
   void upper();


   /** Bufferize an UTF-8 string.

      This is an efficient shortcut for the very common case of UTF8 strings
      being turned into falcon string.
      There isn't a drect constructor that understands that the input char *
      is an UTF8 string, but the proxy generators UTF8String and UTF8String
      serve to this purpose.

      After the call, the previous content of this string is destroyed.

      In case of an invalid UTF8 sequence, up to what is possible to decode is
      read, and the function return false.

      \param utf8 the utf8 string to be loaded
      \param len Expected length (-1 to scan for '\0' in the input string).
      \return true on success, false if the sequence is invalid.

   */
   bool fromUTF8( const char *utf8, length_t len );

   bool fromUTF8( const char *utf8 );

   /** Determines the length in characters of the given UTF string.
    * \param utf8 A utf8 character sequence.
    * \param len Size of the sequence, or String::npos to proceed up to a zero character.
    * \return a length, or String::npos if the input is not a valid utf8 sequence.
    */
   static length_t UTF8Size( const char *utf8, length_t len = String::npos );

   /** Access to a single character.
      Please, notice that Falcon strings are polymorphic in assignment,
      so they cannot support the following syntax:
      \code
         s[n] = c; // can't work with Falcon strings.
      \endcode

      This operator is provided as a candy grammar for getCharAt().
   */
   char_t operator []( length_t pos ) const { return getCharAt( pos ); }

   /** Adds an extra '\0' terminator past the end of the string.

      This makes the string data (available through getRawStorage()) suitable
      to be sent to C functions compatible with the character size of this
      string.

      Eventually, it should be preceded by a call to setCharSize().
   */
   const char* c_ize() const;

   /** Compares a string with the beginning of this string.
      If \b str is empty, returns true, if it's larger than
      this string returns false.
      \param str The string to be compared against the beginning of this string.
      \param icase true to perform a case neutral compare
      \return true on success.
   */
   bool startsWith( const String &str, bool icase=false ) const;

   /** Compares a string with the end of this string.
      If \b str is empty, returns true, if it's larger than
      this string returns false.
      \param str The string to be compared against the end of this string.
      \param icase true to perform a case neutral compare
      \return true on success.
   */
   bool endsWith( const String &str, bool icase=false ) const;

   /** Matches this string against a dos-like wildcard.
      \param wildcard A dos-like wildcard.
      \param bICase true if this function should ignore the character case of the two strings.
      \return true if the wildcard matches this string.
   */
   bool wildcardMatch( const String& wildcard, bool bICase = false ) const;

   /** Makes all the quotes and double quotes in this string to be preceeded by a '\' slash */
   void escapeQuotes();

   /** Removes all the slashes before quotes. */
   void unescapeQuotes();

   /** Replaces all the occurences of a substring with the given needle */
   void replace( const String& needle, const String& repl, String& target, int32 count = -1) const;

   /** Alters the character size of this string.

       Changes the number of bytes used to represent a single
       character in this string. The number of byte used can be
       1, 2 or 4.

       If the original character size was different, the
       string is bufferized into a new memory area, otherwise
       the string is unchanged.

       @param nsize The new character size for the string.
       @return True if the nsize value is valid, false otherwise.
   */
   bool setCharSize( uint16 nsize );

   /** Returns true if the given character is a whitespace. */
   inline static bool isWhiteSpace( char_t chr )
   {
      return chr == ' ' || chr == '\t' || chr == '\r' || chr == '\n';
   }
   
   void gcMark( uint32 mark ) { m_lastMark = mark; }
   uint32 currentMark() const { return m_lastMark; }
   String* clone() const { return new String(*this); }

   static Class* m_class_handler;

   bool isImmutable() const { return m_bImmutable; }
   void setImmutable( bool i ) {  m_bImmutable = i; }

   //===========================================================
   // Type checking
   //

   /** Check if this string is entirely composed of alphabet characters.
       \return true if this string is entirely composed of alphabet characters.
   */
   bool isAlpha() const;

   /** Check if this string is entirely composed of 0-9 Arabic digits.
       \return true if this string is entirely composed of 0-9 Arabic digits.
   */
   bool isDigit() const;

   /** Check if this string is entirely composed of alphabet characters or 0-9 digits.
       \return true if this string is entirely composed of alphabet characters or 0-9 digits.
   */
   bool isAlphaNum() const;

   /** Check if this string is entirely composed of punctation characters.
       \return true if this string is entirely composed of punctation characters.
   */
   bool isPunct() const;

   /** Check if this string is entirely composed of upper case Latin alphabet characters.
       \return true if this string is entirely composed of alphabet characters.
   */
   bool isUpper() const;


   /** Check if this string is entirely composed of lower case Latin alphabet characters.
       \return true if this string is entirely composed of alphabet characters.
   */
   bool isLower() const;

   /** Check if this string is entirely composed of space, tabulation, newline or return.
       \return true if this string is entirely composed of space, tabulation, newline or return.
   */
   bool isWhitespace() const;

   /** Check if this string is in the printable range.
       \return true if his string is in the printable range.
   */
   bool isPrintable() const;

   /** Check if this string is entirely composed of characters in the range 0-127.
       \return true if this string is entirely composed of characters in the range 0-127.
   */
   bool isASCII() const;

   /** Check if this string is entirely composed of characters in the range 0-255.
       \return true if this string is entirely composed of characters in the range 0-255.
   */
   bool isISO() const;

   /** Check if the given character is an alphabet character.
       \return true if the given character is an alphabet character.
   */
   static bool isAlpha( uint32 chr );

   /** Check if the given character is one of the 0-9 Arabic digits.
       \return true the given character is one of the 0-9 Arabic digit.
   */
   static bool isDigit( uint32 chr );

   /** Check if the given character is an alphabet character or a digit.
       \return true if the given character is an alphabet character or a digit.
   */
   static bool isAlphaNum( uint32 chr );

   /** Check if the given character is an alphabet character.
       \return true if the given character is an alphabet character.
   */
   static bool isPunct( uint32 chr );

   /** Check if the given character is an upper case Latin alphabet character.
       \return true if the given character is an upper case Latin alphabet character.
   */
   static bool isUpper( uint32 chr );


   /** Check if the given character is a lower case Latin alphabet character.
       \return true if the given character is a lower case Latin alphabet character.
   */
   static bool isLower( uint32 chr );

   /** Check if the given character is one of space, tabulation, newline or return.
       \return true if the given character is one of space, tabulation, newline or return.
   */
   static bool isWhitespace( uint32 chr );

   /** Check if the given character is in the printable range.
       \return true if the given character is in the printable range.
   */
   static bool isPrintable( uint32 chr );

   /** Check if the given character is in the range 0-1const27.
       \return true if the given character is in the range 0-127.
   */
   static bool isASCII( uint32 chr );

   /** Check if the given character is in the range 0-255.
       \return true if the given character is in the range 0-255.
   */
   static bool isISO( uint32 chr );

private:
   class TrimCheckerSet;
   class TrimCheckerWS;

   class CC_Alpha;
   class CC_Digit;
   class CC_AlphaNum;
   class CC_Punct;
   class CC_Whitespace;
   class CC_Upper;
   class CC_Lower;
   class CC_Printable;
   class CC_ASCII;
   class CC_ISO;

   template<class __Checker> void trimInternal( String::t_trimmode mode, const __Checker& checker );

   template<class __Checker> bool checkTypeInternal( const __Checker& checker ) const;
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


length_t csh::Base::length(const String* str) const
{
    return str->size()/m_charSize; 
}

}

#endif

/* end of string.h */

