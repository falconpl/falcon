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

#ifndef FALCON_WOPI_REPLY_STREAM_H
#define FALCON_WOPI_REPLY_STREAM_H

#include <falcon/stream.h>
#include <falcon/multiplex.h>

namespace Falcon {
namespace WOPI {

class Reply;

/**
 This is a procy stream sensing for the first output
   operation and then invoking the Reply for commit.
*/

class ReplyStream: public Stream
{
public:
   ReplyStream( Reply* rep , Stream* underlying, bool bMakeCH = true );
   ReplyStream( const ReplyStream& other );
   ~ReplyStream();

   virtual size_t read( void *buffer, size_t size );
   virtual size_t write( const void *buffer, size_t size );
   virtual bool close();
   virtual int64 tell();
   virtual bool truncate( off_t pos=-1 );
   virtual off_t seek( off_t pos, e_whence w );
   virtual bool flush();
   virtual Stream *clone() const;

   /** We'll return the disk (neutral) engine standard factory */
   virtual const Multiplex::Factory* multiplexFactory() const;

private:

   Reply* m_rep;
   Stream* m_underlying;
};


}
}

#endif

/* end of replystream.h */

