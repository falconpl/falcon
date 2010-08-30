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

namespace Falcon {
namespace WOPI {

ReplyStream::ReplyStream( Reply* rep ):
      Stream( t_proxy ),
      m_rep( rep )
{
}

ReplyStream::ReplyStream( const ReplyStream& other ):
      Stream( other ),
      m_rep( other.m_rep )
{
}

ReplyStream::~ReplyStream()
{
}

bool ReplyStream::writeString( const String &source, uint32 begin, uint32 end )
{
   if ( ! m_rep->isCommited() )
      m_rep->commit();

   return m_rep->output()->writeString( source, begin, end );
}

bool ReplyStream::close()
{
   if ( ! m_rep->isCommited() )
      m_rep->commit();

   return m_rep->output()->close();
}

int32 ReplyStream::write( const void *buffer, int32 size )
{
   if ( ! m_rep->isCommited() )
      m_rep->commit();

   return m_rep->output()->write( buffer, size );
}

int ReplyStream::writeAvailable( int n, const Sys::SystemData* sd )
{
   return m_rep->output() ?  1 :
          m_rep->output()->writeAvailable( n, sd );
}

int64 ReplyStream::lastError() const
{
   return m_rep->output()->lastError();
}

bool ReplyStream::put( uint32 chr )
{
   if ( ! m_rep->isCommited() )
      m_rep->commit();

   return m_rep->output()->put( chr );
}


bool ReplyStream::get( uint32 &chr )
{
   m_status = t_unsupported;
   return false;
}

Stream *ReplyStream::clone() const
{
   return new ReplyStream( *this );
}

// Flushes the stream.
bool ReplyStream::flush()
{
   if ( ! m_rep->isCommited() )
      m_rep->commit();

   return m_rep->output()->flush();
}


}
}

/* end of replystream.cpp */
