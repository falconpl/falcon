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
   int32 m_bufSize;
   bool m_changed;
   byte *m_buffer;
   int32 m_bufPos;
   int32 m_bufLen;

   uint64 m_filePos;
   bool m_bReseek;

   Stream *m_stream;
   bool m_streamOwner;

   bool refill();

protected:
   virtual int64 seek( int64 pos, e_whence whence );
   
public:
   StreamBuffer( Stream *underlying, bool bOwn = true, uint32 bufSize = default_buffer_size );
   StreamBuffer( const StreamBuffer &other );
   virtual ~StreamBuffer();
   
   virtual StreamBuffer *clone() const;
   virtual bool isStreamBuffer() const { return true; }
   

   /** Returns the underlying stream used by this transcoder.
   \return the underlying stream.
   */
   Stream *underlying() const { return m_stream; }

   virtual bool close();
   virtual int64 tell();
   virtual bool truncate( int64 pos=-1 );
   virtual int32 readAvailable( int32 msecs_timeout, const Sys::SystemData *sysData = 0 );
   virtual int32 writeAvailable( int32 msecs_timeout, const Sys::SystemData *sysData );
   virtual bool flush();
   
   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual int32 read( void *buffer, int32 size );
   virtual int32 write( const void *buffer, int32 size );

   virtual bool errorDescription( ::Falcon::String &description ) const {
      return m_stream->errorDescription( description );
   }
   virtual int64 lastError() const { return m_stream->lastError(); }
   virtual t_status status() const { return m_stream->status(); }
   virtual void status(t_status s) { return m_stream->status(s); }
   
   /** Disengages this transcoder from the underlying stream. */
   void detach() { m_stream = 0; m_streamOwner = false; }
   
   bool resizeBuffer( uint32 size );
   uint32 bufferSize() const { return m_bufSize; }
};

}

#endif

/* end of streambuffer.h */
