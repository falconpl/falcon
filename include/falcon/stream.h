/*
   FALCON - The Falcon Programming Language.
   FILE: file_base.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 2 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The Falcon Stream API.

   Falcon Streams are the basic I/O of every Falcon subsystem, including
   VM, compiler, assembler, generators and modules.
*/

#ifndef flc_stream_H
#define flc_stream_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/falcondata.h>
#include <falcon/string.h>
#include <falcon/vm_sys.h>

#define FALCON_READAHEAD_BUFFER_BLOCK  32

namespace Falcon {


/** Base class for file and filelike services.

   This class is used by all the I/O in Falcon libraries and modules.

   Subclassess to store or read data from standard streams, files and
   memory buffers are already provided; the implementors may extend
   this class or the derived classes to support more systems and/or
   special I/O devices.

   This is a purely abstract class that serves as a base class for
   system-specific or system independent implementations.
*/

class FALCON_DYN_CLASS Stream: public FalconData
{
protected:
   uint32 *m_rhBuffer;
   uint32 m_rhBufferSize;
   uint32 m_rhBufferPos;

   /** Push a character in the read ahead buffer.
      If the buffer is not still constructed, it is
      constructed here.
      \param chr the char to be pushed.
   */
   void pushBuffer( uint32 chr );

   /** Pops next character from the buffer.
      If the buffer is not still constructed, or if its empty,
      the functionr returns false.
      \param chr the character that will be retreived.
      \return false if buffer empty.
   */
   bool popBuffer( uint32 &chr );

   /** Returns true if the buffer is empty. */
   bool bufferEmpty() const { return m_rhBufferPos == 0; }

   /** Protected constructor.
      Transcoder constructor of this class and of all subclassess
      is guaranteed not to use the stream in any way
      but to store it in a protected member; so it is actually possible to
      switch streams after creation, or to pass 0 as stream initially.
      Just, be sure YOU KNOW WHAT YOU ARE DOING, as switching streams on
      stateful encoders may be a bad idea, as it may be a bad idea to
      create a transcoder passing a 0 stream to it.

      Also, there's nowhere any control about the stream not being
      null before calling parsing functions, so be carefull.
   */

public:
   typedef enum {
      t_undefined = 0,
      t_file = 1,
      t_stream = 2,
      t_membuf = 3,
      t_network = 4,
      t_proxy = 5
   } t_streamType;

   typedef enum {
      t_none = 0,
      t_open = 0x1,
      t_eof = 0x2,
      t_error = 0x4,
      t_unsupported = 0x8,
      t_invalid = 0x10,
      t_interrupted = 0x20

   } t_status ;


protected:
   t_streamType m_streamType;
   t_status m_status;
   int32 m_lastMoved;

   /** Initializes the base file class. */
   Stream( t_streamType streamType ):
      m_rhBuffer( 0 ),
      m_rhBufferSize( 0 ),
      m_rhBufferPos( 0 ),
      m_streamType( streamType ),
      m_status( t_none ),
      m_lastMoved( 0 )
   {}

   typedef enum {
      ew_begin,
      ew_cur,
      ew_end
   } e_whence;

   friend class Transcoder;

public:

   Stream( const Stream &other );

   t_streamType type() const { return m_streamType; }
   virtual t_status status() const { return m_status; }
   virtual void status( t_status s ) { m_status = s; }

   uint32 lastMoved() const { return m_lastMoved; }

   void reset();
   bool good() const;
   bool bad() const;
   bool open() const;
   bool eof() const;
   bool unsupported() const;
   bool invalid() const;
   bool error() const;
   bool interrupted() const;

   virtual ~Stream();

   virtual void gcMark( uint32 mark ) {}
   virtual bool isStreamBuffer() const { return false; }
   virtual bool isTranscoder() const { return false; }

   /** Reads from target stream.

      \param buffer the buffer where read data will be stored.
      \param size the amount of bytes to read
   */
   virtual int32 read( void *buffer, int32 size );

   /** Write to the target stream.
   */
   virtual int32 write( const void *buffer, int32 size );

   /** Close target stream.
   */
   virtual bool close();
   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );
   virtual bool errorDescription( ::Falcon::String &description ) const;

   /** Determines if the stream can be read, possibly with a given timeout.
      If sysData is not zero, it will be used to honor concurrent interrupt requests.
   */
   virtual int32 readAvailable( int32 msecs_timeout, const Sys::SystemData *sysData = 0 );

   /** Determines if the stream can be written, possibly with a given timeout.
      If sysData is not zero, it will be used to honor concurrent interrupt requests.
   */
   virtual int32 writeAvailable( int32 msecs_timeout, const Sys::SystemData *sysData = 0 );

   int64 seekBegin( int64 pos ) {
      return seek( pos, ew_begin );
   }

   int64 seekCurrent( int64 pos ) {
      return seek( pos, ew_cur );
   }

   int64 seekEnd( int64 pos ) {
      return seek( pos, ew_end );
   }
   
   virtual int64 seek( int64 pos, e_whence w );

   virtual int64 lastError() const;

   /** Gets next character from the stream.
      Subclasses must manage both stateful transcoding and
      properly popping readahead characters from the buffer.
      \return true if the character is available, false on stream end or error.
   */
   virtual bool get( uint32 &chr ) = 0;

   /** Gets a whole string from the stream.
      This is implemented by iteratively calling get( uint32 ).
      The caller should provide a string with enough space already reserved,
      if possible, to make operations more efficient.

      The target string may be shorter than required if the stream ends before all the
      characters are read, or in case of error.

      \return true if some characters are available, false on stream end or error.
   */
   virtual bool readString( String &target, uint32 size );

   /** Writes a character on the stream.
      \param chr the character to write.
      \return true success, false on stream error.
   */
   virtual bool put( uint32 chr );

   /** Writes a string on the stream.
      Encoding range is in [begin, end), that is, the last character encoded is end - 1.
      \param source the string that must be encoded
      \param begin first character from which to encode
      \param end one past last character to encode (can be safely greater than string lenght() )
      \return true success, false on stream error.
   */
   virtual bool writeString( const String &source, uint32 begin=0, uint32 end = csh::npos );

   /** Write a character in the readahead buffer.
      Next get() operation will return characters pushed in the buffer in
      reverse order. So, if the next character on the stream is 100 and the
      caller class unget(1) and unget(2), three consecutive get will
      return in turn 2, 1 and 100.

      Unget is interleaved with readAhead(), so that the sequence
      \code
         Stream *s = ...
         Transcoder x( s );
         x.unget( 10 );
         x.readAhead( chr ); // chr <- 20
         x.unget( 30 );

         String res;
         x.get( res, 3 );
      \endcode

      Will fill res with 30, 20 and 10.
      \see readAhead
      \param chr the character to be pushed.
   */
   void unget( uint32 chr ) { pushBuffer( chr ); }

   /** Ungets a whole string.
      The string is pushed on the read back buffer so that the
      next target.length() get() operations return the content
      of the string.

      \note use wisely.
   */
   void unget( const String &target );

   /** Read a character but don't remove from get().
      This function is equivalent to:
      \code
         Stream *s = ...
         Transcoder xss );
         uint32 chr;
         x.get( chr );
         x.unget( chr );
      \endcode
      \param chr the read character
      \return false on stream end or error.
   */

   bool readAhead( uint32 &chr );

   /** Read a string but don't remove from get().
      Every character in the returned string will still be read by other
      get() operations; this allows to "peek" forward a bit in the
      target stream to i.e. take lexer decisions that won 't affect
      a parser.

      The target string may be shorter than required if the stream ends before all the
      characters are read, or in case of error.

      \param target the read string
      \param size the amount of character to be read.
      \return false on stream end or error.
   */
   bool readAhead( String &target, uint32 size );

   /** Discards ungetted and read ahead characters.
      If the lexer finds that it would be useless to retreive again the
      read ahead characters, it can use this function to discard the content
      of the buffer instead of re-reading and ignoring them.

      However, this can be done only if the final application has a state
      memory of what is happening, as there may be some ungetted or
      read ahaead strings that the code portion calling this function
      may not be aware of.

      In that case, the caller should know the amount of character it
      has read ahead and pass as parameter for this function.

      \param count number of character to discard from read ahead buffer (0 for all).
   */
   void discardReadAhead( uint32 count = 0 );

   /** Flushes stream buffers.
      Hook for buffered streams.
   */
   virtual bool flush();

   /** Clones the stream.
      This version returns 0 and sets error to unsupported;
      subclasses must properly clone the stream.
   */
   virtual Stream *clone() const;
};


/** Or operator on status bitfiled.
   This is to allow integer oinline processing on enum fields in Stream class.
*/
inline Stream::t_status operator|( const Stream::t_status &elem1, const Stream::t_status &elem2)
{
   return static_cast<Stream::t_status>(
      static_cast<unsigned int>(elem1) | static_cast<unsigned int>(elem2) );
}

/** And operator on status bitfiled.
   This is to allow integer oinline processing on enum fields in Stream class.
*/
inline Stream::t_status operator&( const Stream::t_status &elem1, const Stream::t_status &elem2)
{
   return static_cast<Stream::t_status>(
      static_cast<unsigned int>(elem1) & static_cast<unsigned int>(elem2) );
}

/** Xor operator on status bitfiled.
   This is to allow integer oinline processing on enum fields in Stream class.
*/

inline Stream::t_status operator^( const Stream::t_status &elem1, const Stream::t_status &elem2)
{
   return static_cast<Stream::t_status>(
      static_cast<unsigned int>(elem1) ^ static_cast<unsigned int>(elem2) );
}

/** Not operator on status bitfiled.
   This is to allow integer oinline processing on enum fields in Stream class.
*/

inline Stream::t_status operator~( const Stream::t_status &elem1 )
{
   return static_cast<Stream::t_status>( ~ static_cast<unsigned int>(elem1) );
}

inline void Stream::reset()
{
   status( status() &
         static_cast<t_status>(~static_cast<unsigned int>(t_error|t_unsupported|t_invalid)) );
   m_lastMoved = 0;
}

inline bool Stream::good() const
      { return (status() &( t_error | t_unsupported | t_invalid )) == 0; }
inline bool Stream::bad() const
      { return (status() &( t_error | t_unsupported | t_invalid )) != 0; }

inline bool Stream::open() const
   { return (status() & t_open ) != 0; }
inline bool Stream::eof() const
   { return (status() & t_eof ) != 0; }
inline bool Stream::unsupported() const
   { return (status() & t_unsupported ) != 0; }
inline bool Stream::invalid() const
   { return (status() & t_invalid ) != 0; }
inline bool Stream::error() const
   { return ( status() & t_error ) != 0; }
inline bool Stream::interrupted() const
   { return ( status() & t_interrupted ) != 0; }

} //end of Falcon namespace

#endif

/* end of file_base.h */
