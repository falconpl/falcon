/*
   FALCON - The Falcon Programming Language.
   FILE: writer.h

   Base abstract class for file writers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 20:46:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_WRITER_H
#define	_FALCON_WRITER_H

#include <falcon/setup.h>
#include <falcon/types.h>


namespace Falcon {

class Stream;

/** Base abstract class for stream writers.

 A Falcon Stream is a very basic, raw representation of an I/O resource. It's
 methods maps directly to the lowest possible level of system direct resource
 handling functions.

 This makes raw access to streams extremely efficient, but more articulated access,
 as the steps require to serialize or de-serialize a resource, might be extremely
 inefficient.

 \see Reader
 */
class FALCON_DYN_CLASS Writer
{
public:
   virtual ~Writer();

   /** Delegates another Writer.
    \param target Another Writer that will be in charge of handling this stream.

    The stream control, and eventually ownership, is passed onto another writer.
    The state of the underlying stream, including its current buffer, is maintained
    coherent and passsed onto the target.

    \note Using this writer after the this call has an undefined behavior.
    */
   void delegate( Writer& target );

   /** Changes the buffer size.

    By default, the I/O buffering is set to exactly the size of a "memory page"
    (4096 bytes on most systems). It is suggested to set this size to a multiple
    of the memory page size.

    */
   virtual void setBufferSize( length_t bs );

   /** Write all the pending data that's left.
    \throw IOError on error in flushing the stream.
    \throw InterruptedError if the I/O operation has been interrupted.
   */
   virtual bool flush();

   /** Writes to the internal buffer and eventually store what's written to a file.
      It is suggested to use directly m_buffer when possible.
   */
   virtual bool write( byte* data, size_t dataSize );

protected:
   /** Create for normal operations. */
   Writer( Stream* stream, bool bOwn = false );

   /** Create for immediate delegation. */
   Writer();

   byte* m_buffer;
   length_t m_bufPos;
   length_t m_bufSize;

protected:
   bool m_bOwnStream;
   Stream* m_stream;
};

}

#endif	/* _FALCON_WRITER_H */

/* end of writer.h */
