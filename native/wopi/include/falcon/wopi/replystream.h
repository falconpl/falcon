/*
   FALCON - The Falcon Programming Language.
   FILE: replystream.h

   Falcon WOPI - Web Oriented Programming Interface

   This is a dummy stream sensing for the first output
   operation and then invoking the Reply for commit.

   The stream is then destroyed and the first write
   operation is invoked on the real stream.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Mar 2010 16:47:02 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef REPLY_STREAM_H
#define REPLY_STREAM_H

#include <falcon/stream.h>

namespace Falcon {
namespace WOPI {

class Reply;

/**
 This is a dummy stream sensing for the first output
   operation and then invoking the Reply for commit.

   The stream is then destroyed and the first write
   operation is invoked on the real stream.
*/

class ReplyStream: public Stream
{
public:
   ReplyStream( Reply* rep );
   ReplyStream( const ReplyStream& other );
   ~ReplyStream();


    // We don't really need to implement all of those;
   // as we want to reimplement output streams, we'll just
   // set "unsupported" where we don't want to provide support.
   bool writeString( const String &source, uint32 begin=0, uint32 end = csh::npos );
   virtual bool close();
   virtual int32 write( const void *buffer, int32 size );
   virtual int32 writeAvailable( int, const Sys::SystemData* );
   virtual int64 lastError() const;
   virtual bool put( uint32 chr );
   virtual bool get( uint32 &chr );
   virtual Stream *clone() const;

   // Flushes the stream.
   virtual bool flush();

private:

   Reply* m_rep;
};


}
}

#endif

/* end of replystream.h */

