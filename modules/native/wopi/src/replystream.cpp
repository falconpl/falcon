/*
   FALCON - The Falcon Programming Language.
   FILE: replystream.cpp

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

#include <falcon/wopi/reply.h>
#include <falcon/wopi/replystream.h>
#include <falcon/stdmpxfactories.h>

namespace Falcon {
namespace WOPI {

ReplyStream::ReplyStream( Reply* rep, Stream* under ):
      m_rep( rep ),
      m_underlying( under )
{
   m_underlying->incref();
}

ReplyStream::ReplyStream( const ReplyStream& other ):
      Stream( other ),
      m_rep( other.m_rep ),
      m_underlying( other.m_underlying )
{
   m_underlying->decref();
}

ReplyStream::~ReplyStream()
{
   m_underlying->decref();
}

bool ReplyStream::close()
{
   m_rep->commit( m_underlying );
   return m_underlying->close();
}

int64 ReplyStream::tell()
{
   return m_underlying->tell();
}

bool ReplyStream::truncate( off_t pos )
{
   return m_underlying->truncate( pos );
}


off_t ReplyStream::seek( off_t pos, e_whence w )
{
   return m_underlying->seek( pos, w );
}

size_t ReplyStream::read( void *buffer, size_t size )
{
   return m_underlying->read( buffer, size );
}

size_t ReplyStream::write( const void *buffer, size_t size )
{
   m_rep->commit( m_underlying );
   return m_underlying->write( buffer, size );
}

Stream *ReplyStream::clone() const
{
   return new ReplyStream( *this );
}

// Flushes the stream.
bool ReplyStream::flush()
{
   m_rep->commit( m_underlying );
   return m_underlying->flush();
}

const Multiplex::Factory* ReplyStream::multiplexFactory() const
{
   return Engine::instance()->stdMpxFactories()->diskFileMpxFact();
}


}
}

/* end of replystream.cpp */
