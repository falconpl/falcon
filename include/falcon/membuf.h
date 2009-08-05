/*
   FALCON - The Falcon Programming Language.
   FILE: membuf.h

   Core memory buffer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 17 Mar 2008 23:07:21 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Memory buffer - Pure memory for Falcon.
*/

#ifndef flc_membuf_H
#define flc_membuf_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <falcon/deepitem.h>
#include <falcon/error.h>

namespace Falcon {

class VMachine;
class Stream;
class CoreObject;

class FALCON_DYN_CLASS MemBuf: public DeepItem, public Garbageable
{
public:
   typedef void (*tf_deletor)( void* memory );
   enum {
      INVALID_MARK = 0xFFFFFFFF,
      MAX_LEN = 0xFFFFFFFF
   } t_enum_mark;

protected:
   byte *m_memory;

   uint32 m_length;
   uint32 m_mark;
   uint32 m_limit;
   uint32 m_position;

   uint16 m_wordSize;
   uint16 m_byteOrder;

   CoreObject *m_dependant;
   tf_deletor m_deletor;

public:

   /** Creates a pure memory based memory buffer.
    *
    * This constructor creates a memory buffer containing a memory data that
    * will be freed at buffer destruction.
    */
   MemBuf( uint32 ws, uint32 length );

   /** Creates an unowned memory buffer.
    *
    * This simply stores the data in this memory buffer;
    * the ownership of the data stays on the creator, which must dispose it
    * separately.
    *
    * \param ws Size of each element in bytes (1-4)
    * \param data The memory buffer
    * \param length Maximum length of the buffer; pass MAX_LEN if unknown, or
    *        0 if unknown and unavailable to the calling program.
    */
   MemBuf( uint32 ws, byte *data, uint32 length );

   /** Creates an owned memory buffer.
    *
    * This stores the memory in the memory buffer and associates it with
    * a deletor function which will be called when this MemBuf is disposed of.
    * the ownership of the data stays on the creator, which must dispose it
    * separately.
    *
    * \param ws Size of each element in bytes (1-4)
    * \param data The memory buffer
    * \param length Maximum length of the buffer; pass MAX_LEN if unknown, or
    *        0 if unknown and unavailable to the calling program.
    * \param deletor The deletor for the given data. Pass memFree for data
    *        created with memAlloc, or 0 to make data unowned.
    */
   MemBuf( uint32 ws, byte *data, uint32 length, tf_deletor deletor );

   /**
    * Destructor.
    *
    * The destructor will also destroy the associated data if a deletor
    * function has been specified in the constructor.
    */
   virtual ~MemBuf();

   uint16 wordSize() const { return m_wordSize; }
   uint32 length() const { return m_length; }

   virtual uint32 get( uint32 pos ) const = 0;
   virtual void set( uint32 pos, uint32 value ) = 0;

   uint32 position() const { return m_position; }
   uint32 getMark() const { return m_mark; }
   uint32 limit() const { return m_limit; }

   /** Puts a value at current position and advance.

      Will throw AccessError on error. This is meant
      to be called from scripts.
   */
   void put( uint32 data )
   {
      if ( m_position >= length() )
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).module( __FILE__ ).symbol( "put" ).extra( "MemBuf" ) );

      set( m_position++, data );
   }

   /** Gets a value at current position and advance.

      Will throw AccessError on error. This is meant
      to be called from scripts.
   */
   uint32 get()
   {
      if ( m_position >= m_limit )
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).module( __FILE__ ).symbol( "get" ).extra( "MemBuf" ) );

      return get( m_position++ );
   }

   /** Returns the size of the buffer in bytes.
      It's the length of the buffer * the word size.
   */
   uint32 size() const { return m_length * m_wordSize; }

   /** Resizes the Memory Buffer
      Invariants are maintained.
      \note This method uses memRealloc, so it can be used only
         if the memory buffer has been created through memAlloc and should
         be released with memFree. The method doesn't perform any check, so
         be sure that you're knowing what you're doing when you use it.

      \param newSize number of items stored.
   */
   void resize( uint32 newSize );

   /** Returns the raw buffer data. */
   byte *data() const { return m_memory; }

   /** Sets the limit of read operations. */
   void limit( uint32 l )
   {
      if ( l > length() )
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).module( __FILE__ ).symbol( "limit" ).extra( "MemBuf" ) );

      m_limit = l;
      if ( l < m_position )
         m_position = l;
   }

   /** Set current read-write position.
      Will throw if the limit is less than the current length().
   */
   void position( uint32 p )
   {
      if ( p > m_limit )
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).module( __FILE__ ).symbol( "position" ).extra( "MemBuf" ) );

      m_position = p;
      if ( m_mark < m_position )
         m_mark = INVALID_MARK;
   }

   /** Mark a position for a subquesent reset.
      A subquesent reset() will bring the position() pointer here.
   */
   void placeMark( uint32 m )
   {
      if ( m > m_position )
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).module( __FILE__ ).symbol( "mark" ).extra( "MemBuf" ) );

      m_mark = m;
   }

   /** Mark current position.
      Records the position() as it is now. A subquesent reset() will bring the position() pointer here.
   */
   void placeMark()
   {
      m_mark = m_position;
   }

   /** Returns the pointer at previous mark.
      Raises an access error if the mark is not set.
   */
   void reset()
   {
      if ( m_mark == INVALID_MARK )
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).module( __FILE__ ).symbol( "reset" ).extra( "MemBuf" ) );

      m_position = m_mark;
   }

   /** Rewinds the membuf
      The position is set to 0 and the mark is invalidated.
   */
   void rewind()
   {
      m_position = 0;
      m_mark = INVALID_MARK;
   }


   /** Rewinds the membuf
      The position is set to 0 and the mark is invalidated. Limit is set to size.
   */
   void clear()
   {
      m_position = 0;
      m_limit = m_length;
      m_mark = INVALID_MARK;
   }

   /** After a write, prepares for a read.
      Useful to parse an incoming buffer or to drop the incoming buffer in another place.
   */
   void flip()
   {
      m_limit = m_position;
      m_position = 0;
      m_mark = INVALID_MARK;
   }

   /** Compacts the buffer and prepares it for a new read.
      Remaining data, if any, is moved to the beginning of the buffer.
      Position is set to the previous limit, and limit is set to length.
      The mark is discarded.
   */
   void compact();

   /** Return the amount of items that can be taken before causing an exception. */
   uint32 remaining() const {
      return m_limit - m_position;
   }

   /** Sets the amount of items in this buffer.
      The actual size in bytes is obtained multiplying this length by the word size.
   */
   void length( uint32 s ) { m_length = s; }
   void setData( byte *data, uint32 length, tf_deletor deletor = 0 );

   /** Return the CoreObject that stores vital data for this mempool.
      \see void dependant( CoreObject *g )
   */
   CoreObject *dependant() const { return m_dependant; }

   /** Links this data to a CoreObject.
      Some object may provide MemBuf properties to access data or data portion
      stored in some part of the user_data they reflect.

      To make this possible and easy, the MemBuf can be given
      a back reference to the object that created it. In this way,
      the object will be granted to stay alive as long as the MemBuf is
      alive.
   */
   void dependant( CoreObject *g ) { m_dependant = g; }

   virtual bool serialize( Stream *stream, bool bLive = false ) const;

   static MemBuf *deserialize( VMachine *vm, Stream *stream );

   /** Creates a membuf with defined wordsize.
      The length parameter is the final element count; it gets multiplied
      by nWordSize.
   */
   static MemBuf *create( int nWordSize, uint32 length );

   virtual MemBuf* clone() const;
   virtual void gcMark( uint32 gen );

   virtual void readProperty( const String &, Item &item );
   virtual void writeProperty( const String &, const Item &item );
   virtual void readIndex( const Item &pos, Item &target );
   virtual void writeIndex( const Item &pos, const Item &target );
};

class FALCON_DYN_CLASS MemBuf_1: public virtual MemBuf
{
public:
   MemBuf_1( uint32 length ):
      MemBuf( 1, length )
   {}

   MemBuf_1( byte *data, uint32 length, tf_deletor deletor = 0 ):
      MemBuf( 1, data, length, deletor )
   {}

   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class FALCON_DYN_CLASS MemBuf_2: public virtual MemBuf
{
public:
   MemBuf_2( uint32 length ):
      MemBuf( 2, length )
   {}

   MemBuf_2( byte *data, uint32 length, tf_deletor deletor = 0 ):
      MemBuf( 2, data, length, deletor )
   {}

   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};

class FALCON_DYN_CLASS MemBuf_3: public virtual MemBuf
{
public:
   MemBuf_3( uint32 length ):
      MemBuf( 3, length )
   {}

   MemBuf_3( byte *data, uint32 length, tf_deletor deletor = 0 ):
      MemBuf( 3, data, length, deletor )
   {}

   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 deletor );
};

class FALCON_DYN_CLASS MemBuf_4: public virtual MemBuf
{
public:
   MemBuf_4( uint32 length ):
      MemBuf( 4, length )
   {}

   MemBuf_4( byte *data, uint32 length, tf_deletor deletor = 0 ):
      MemBuf( 4, data, length, deletor )
   {}

   virtual uint32 get( uint32 pos ) const;
   virtual void set( uint32 pos, uint32 value );
};


}

#endif

/* end of membuf.h */
