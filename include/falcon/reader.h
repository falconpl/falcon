/*
   FALCON - The Falcon Programming Language.
   FILE: reader.h

   Base abstract class for file writers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_READER_H
#define _FALCON_READER_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/refcounter.h>


namespace Falcon {

class Stream;

/** Base abstract class for stream readers.

 A Falcon Stream is a very basic, raw representation of an I/O resource. It's
 methods maps directly to the lowest possible level of system direct resource
 handling functions.

 This makes raw access to streams extremely efficient, but more articulated access,
 as the steps require to serialize or de-serialize a resource, might be extremely
 inefficient.

 So, Falcon exposes a Streaming Framework composed of a series of helper classes
 which allow efficient access to raw streams under complex usage patterns.

 The framework is composed of four class hierarcies, based on the following
 base classes:
 - Stream: abstract representation of a system I/O resource; files, sockets, pipes
   and process streams are some example.
 - StreamBuffer: wrapper class providing buffered random I/O for an underlying
   stream. StreamBuffer is a concrete (non-abstract) class.
 - Reader: Facility to read a particular kind of data from a stream. TextReader
   and DataReader are the most prominent concrete subclasses.
 - Writer: Facility to write a particular kind of data to a stream. TextWriter
   and DataWriter are the most prominent concrete sublasses.

 Using one of the Streaming Framework class to read from or write to a stream is
 called "delegate the stream"; delegating happens in one verse, so you can delegate
 read, write or both. Once delegated, using the stream directly has undefined
 result, as the delegate has its own buffering and read-ahead strategies, that
 might leave the underlying stream in an undefined state.

 However, the delegated stream state is coherent inside a hierarcy, so it is possible
 to change the type of reader, writer or buffer multiple times.

 It is possible to delegate full duplex streams (like socket or pipes) both to
 readers and writers at the same time; trying to delegate a random access stream
 to both a reader and a writer has an undefined result (being read and write buffers
 separate and non-communicating).

 The StreamBuffer hierarcy is explicitly provided to handle random access streams
 efficienctly with higher granularity with respect to the raw system interface
 offered by the Stream classes.

 A stream might be owned by its delegate or not. If owned, it is also closed and
 deleted at delegate destruction. When a delegate sub-delegates its stream, the
 ownership is passed as well, and the stream is closed and destroyed after
 the destruction of the new delegate only.
 */
class FALCON_DYN_CLASS Reader
{
public:

   /** Delegates another Reader.
    \param target Another Reader that will be in charge of handling this stream.

    The stream control, and eventually ownership, is passed onto another reader.
    The state of the underlying stream, including its current buffer, is maintained
    coherent and passsed onto the target.

    \note Using this reader after the this call has an undefined behavior.
    */
   void delegate( Reader& target );

   /** Changes the "suggested" buffer size.

    Unless the Reader subclass has a different policy, this is the size of the
    blocks that shall be read each time.

    Internally, the size of the buffer is the twice as requested, so that it is
    always possible to read "more" buffer_size bytes from the underlying stream
    even when buffer_size-1 bytes are still unprocessed. This allows to always use
    the same size in requests to the underlying O/S, which maximizes the raw I/O
    performance.

    So, this becomes the suggested size of the I/O reads, while the memory employed
    by this class is usually the twice of it.

    By default, the I/O buffering is set to exactly the size of a "memory page"
    (4096 bytes on most systems). It is suggested to set this size to a multiple
    of the memory page size.

    */
   virtual void setBufferSize( length_t bs );

   /** Changes the underlying stream.
    \param s The new stream that this reader should read from.
    \param bOwn if true, the stream is owned by this Reader (and destroyed at Reader destruction).
    \param bDiscard Discard the buffered data still unread coming from the old stream.

    Pending reads on the previous stream are maitanied (the read buffer is not emptuy),
    so it may take several reads before the contents of the new streams are actually
    fetched. This behavior can be overridden by setting bDiscard to true; in this case
    any data fetched from the old stream but still unread is discarded.

    If it was owned, the previous stream is destroyed.
    */
   virtual void changeStream( Stream* s, bool bDiscard = false );
      
   /** Returns true if the underlying stream is exhausted. */
   virtual bool eof() const;
   
   /** Discards read buffer and syncs with current position in the underlying stream. */
   void sync();
   
protected:
   /** Create for normal operations. */
   Reader( Stream* stream );

   Reader( const Reader& other );
   /** Create for immediate delegation */
   Reader();
   
   virtual ~Reader();


   /** Refills the read buffer with new data. */
   virtual bool refill();   

   /** Chech the read buffer and eventually refills it with new data.
    \param suggestedSize try to read this size (or possibly more) from the
    next read.
    \throw IOError in case of read error.
    \return true If the required bytes are available, false if the stream is at eof.

    Fetch will return true while there is still unread data in the buffer, and if
    possible, it will try to get in the buffer at least suggestedSize bytes from
    the stream (in one single read).

    This method is similar to ensure(), but it doesn't force the required size to
    be read before returning. Operations performed by fetch are the following:
    
    - If the buffer has still more than suggestedSize to be read, return immediately true.
    - If the stream is at EOF, return true if there is still something to read.
    - If the read size is smaller than the suggested size, resize the buffer to
      accomodate suggestedSize bytes.
    - refill the buffer and return true.

    This method is useful when variable length data is to be read from the stream,
    its size isn't know in advance, and reading a smaller amount of data
    can't be considered an error without checking other conditions.
    
   */
   virtual bool fetch( length_t suggestedSize );

   /** Makes sure that there is some data to be read.
    \param size the size of bytes that shall be available from the stream.
    \throw IOError in case of read error.
    \return true If the required bytes are available, false if the stream can't
    provide the required data.

    This method blocks until at least \b size bytes are read. If the required
    size is larger than the current buffer size, it is resized to accomodate
    for the needed data.

    If not enough data is available, more data is read repeatedly until the
    required data is fetched from the stream. If the stream is closed before
    the data can be retrieved, or if the read hits eof, the method return false.

    This method is useful to implement network-oriented protocols or check for
    complete headers being safely stored on files. Using this method is possible
    to abstract the way that a certain header data is fetched into memory and
    becomes available for higher level parsing, having a first failure check
    level if the stream can't provide enough data.
    */
   virtual bool ensure( length_t size );
   
   byte* m_buffer;
   length_t m_bufPos;
   length_t m_bufLength;
   length_t m_bufSize;
   length_t m_readSize;

protected:
   Stream* m_stream;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(Reader);
};

}

#endif	/* _FALCON_TEXTREADER_H */

/* end of reader.h */
