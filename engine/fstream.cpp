/*
   FALCON - The Falcon Programming Language.
   FILE: fstream.cpp

   System independent part of fstream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Mar 2011 20:06:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/fstream.h>
#include <falcon/stdstreamtraits.h>

namespace Falcon{

class Interrupt;

//============================================================
// Base fstream
//

FStream *FStream::clone() const
{
   FStream *ge = new FStream( *this );
   return ge;
}



StreamTraits* FStream::traits() const
{
   static StreamTraits* gen = Engine::streamTraits()->diskFileTraits();
   return gen;
}



FStream::Traits::~Traits()
{}

Multiplex* FStream::Traits::multiplex( Selector* master )
{
   return new FStream::MPX(master);
}


//====================================================================================
//
//====================================================================================

StringStream::MPX::MPX( MultiplexGenerator* generator, Selector* master ):
         Multiplex( generator, master )
{
}

StringStream::MPX::~MPX()
{
}


void StringStream::MPX::addStream( Stream* stream, int mode )
{
   StringStream* ss = static_cast<StringStream*>(stream);

   if( (mode & Selector::mode_write) != 0)
   {
      // always writeable
      onReadyWrite(stream);
   }

   if( (mode & Selector::mode_read) != 0)
   {
      ss->m_b->m_mtx.lock();
      uint32 bsize =  ss->m_b->m_str->size();
      if ( bsize > ss->m_posRead )
      {
         ss->m_b->m_mtx.unlock();
         onReadyRead( stream );
         stream->decref();
         return;
      }

      bool bNew = ss->m_b->m_waiters.insert( this ).second;
      ss->m_b->m_mtx.unlock();

      if( bNew )
      {
         incref();
      }
   }
}


void StringStream::MPX::removeStream( Stream* stream )
{
   StringStream* ss = static_cast<StringStream*>(stream);

   ss->m_b->m_mtx.unlock();
   bool bRemoved = ss->m_b->m_waiters.erase(this) > 0;
   ss->m_b->m_mtx.unlock();

   if( bRemoved )
   {
      decref();
   }
}

void StringStream::MPX::onStringStreamReady( StringStream* ss )
{
   onReadyRead( ss );
   decref();
}

//============================================================
// Input only FStream
//

/** File stream with output functions filtered out. */
size_t InputOnlyFStream::writeAvailable( int32 )
{
   throwUnsupported();
   return 0;
}

size_t InputOnlyFStream::write( const void*, size_t )
{
   throwUnsupported();
   return 0;
}

bool InputOnlyFStream::truncate(off_t)
{
   throwUnsupported();
   return false;
}

InputOnlyFStream* InputOnlyFStream::clone() const
{
   return new InputOnlyFStream(*this);
}

//============================================================
// Ouptut only FStream
//


size_t OutputOnlyFStream::readAvailable( int32 )
{
   throwUnsupported();
   return 0;
}


size_t OutputOnlyFStream::read( void *, size_t )
{
   throwUnsupported();
   return 0;
}


OutputOnlyFStream* OutputOnlyFStream::clone() const
{
   return new OutputOnlyFStream( *this );
}


//============================================================
// Read only FStream
//

off_t ReadOnlyFStream::seek( off_t, Stream::e_whence )
{
   throwUnsupported();
   return 0;
}


off_t ReadOnlyFStream::tell()
{
   throwUnsupported();
   return 0;
}


ReadOnlyFStream* ReadOnlyFStream::clone() const
{
   return new ReadOnlyFStream( *this );
}

//============================================================
// Write only FStream
//


off_t WriteOnlyFStream::seek( off_t, Stream::e_whence )
{
   throwUnsupported();
   return 0;
}

off_t WriteOnlyFStream::tell()
{
   throwUnsupported();
   return 0;
}

bool WriteOnlyFStream::truncate(off_t )
{
   throwUnsupported();
   return 0;
}


WriteOnlyFStream* WriteOnlyFStream::clone() const
{
   return new WriteOnlyFStream(*this);
}

}

/* end of fstream.cpp */
