/*
   FALCON - The Falcon Programming Language.
   FILE: uploaded.cpp

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 16 Oct 2013 15:41:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/uploaded.h>
#include <falcon/engine.h>
#include <falcon/stream.h>
#include <falcon/vfsiface.h>
#include <falcon/gclock.h>
#include <falcon/stringstream.h>

namespace Falcon {
namespace WOPI {

Uploaded::Uploaded():
   m_size(0),
   m_gcMark(0)
{
   m_data = 0;
   m_dataLock = 0;
}

Uploaded::Uploaded( const String& fname, const String& mime, int64 size ):
   m_size(size),
   m_filename( fname ),
   m_mimeType( mime ),
   m_gcMark(0)
{
   m_data = 0;
   m_dataLock = 0;
}

Uploaded::Uploaded( const String& fname, const String& mime, int64 size, const String& storage ):
   m_size(size),
   m_filename( fname ),
   m_mimeType( mime ),
   m_storage( storage ),
   m_gcMark(0)
{
   m_data = 0;
   m_dataLock = 0;
}

Uploaded::~Uploaded()
{
   if( m_dataLock != 0 )
   {
      m_dataLock->dispose();
   }
}


void Uploaded::data( String* value )
{
   if( m_dataLock != 0 )
   {
      m_dataLock->dispose();
   }

   m_data = value;
   m_dataLock = Engine::collector()->lock( FALCON_GC_HANDLE(value) );
}


void Uploaded::read()
{
   byte buffer[4096];

   if ( m_data != 0 )
   {
      // already in
      return;
   }

   LocalRef<Stream> stream( Engine::instance()->vfs().open( m_storage, VFSIface::OParams().rdOnly() ) );
   m_data = new String;

   m_data->reserve( (length_t) m_size );
   m_dataLock = Engine::collector()->lock(FALCON_GC_HANDLE(m_data));

   while( ! stream->eof() )
   {
      size_t count = stream->read( buffer, sizeof(buffer) );
      m_data->append( String().adoptMemBuf( buffer, count, 0 ) );
   }

   m_data->toMemBuf();
   stream->close();
}


Stream* Uploaded::open()
{
   if ( m_data != 0 )
   {
      StringStream* ss = new StringStream(*m_data);
      return ss;
   }

   Stream* stream = Engine::instance()->vfs().open( m_storage, VFSIface::OParams().rdOnly() );
   return stream;
}


void Uploaded::store( const String& target )
{
   Stream* input;
   byte buffer[4096];

   if ( m_data != 0 )
   {
      input = new StringStream(*m_data);
   }
   else {
      input = Engine::instance()->vfs().open( m_storage, VFSIface::OParams().rdOnly().shNoWrite() );
   }

   try
   {
      LocalRef<Stream> output(Engine::instance()->vfs().open( target, VFSIface::CParams().wrOnly().truncate().shNoWrite() ));
      while( ! input->eof() )
      {
         size_t count = input->read( buffer, sizeof(buffer) );
         output->write( buffer, count );
      }

      output->close();
      input->decref();
   }
   catch( ... )
   {
      input->decref();
      throw;
   }
}



void Uploaded::store( Stream* target )
{
   Stream* input;
   byte buffer[4096];

   if ( m_data != 0 )
   {
      input = new StringStream(*m_data);
   }
   else {
      input = Engine::instance()->vfs().open( m_storage, VFSIface::OParams().rdOnly().shNoWrite() );
   }

   try
   {
      while( ! input->eof() )
      {
         size_t count = input->read( buffer, sizeof(buffer) );
         target->write( buffer, count );
      }

      input->close();
      input->decref();
   }
   catch( ... )
   {
      input->decref();
      throw;
   }
}

}
}

/* end of uploaded.cpp */

