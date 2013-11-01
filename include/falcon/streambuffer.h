/*
   FALCON - The Falcon Programming Language.
   FILE: streambuffer.h

   Buffer for stream operations.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 02 Feb 2009 16:26:17 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Buffered stream.
*/

#ifndef FLC_STREAM_BUFFER_H
#define FLC_STREAM_BUFFER_H

#include <falcon/stream.h>

namespace Falcon {

/** Buffered version of basic stream.
   \TODO Everything except reading straight is to be tested.
*/
class FALCON_DYN_CLASS StreamBuffer: public Stream
{
public:
   enum {
         default_buffer_size = 4096
   } enum_default_buffer_size;

private:
   uint32 m_bufSize;
   bool m_changed;
   byte *m_rbuffer;
   byte *m_wbuffer;

   uint32 m_rBufPos;
   uint32 m_rBufLen;

   uint32 m_wBufPos;
   uint32 m_wBufLen;

   uint64 m_filePos;
   bool m_bReseek;

   Stream *m_stream;

   bool refill();

protected:
   virtual int64 seek( int64 pos, e_whence whence );
   
public:
   StreamBuffer( Stream *underlying, uint32 bufSize = default_buffer_size );
   StreamBuffer( const StreamBuffer &other );
   virtual ~StreamBuffer();
   
   virtual StreamBuffer *clone() const;
   virtual bool isStreamBuffer() const { return true; }

   /** Returns the underlying stream used by this transcoder.
   \return the underlying stream.
   */
   virtual Stream *underlying() const { return m_stream; }

   virtual bool close();
   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );
   virtual bool flush();
   
   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual size_t read( void *buffer, size_t size );
   virtual size_t write( const void *buffer, size_t size );

   virtual size_t lastError() const { return m_stream->lastError(); }
   virtual t_status status() const { return m_stream->status(); }
   virtual void status(t_status s) { return m_stream->status(s); }
   
   bool resizeBuffer( uint32 size );
   uint32 bufferSize() const { return m_bufSize; }

   const Multiplex::Factory* multiplexFactory() const;
};

}

#endif

/* end of streambuffer.h */
