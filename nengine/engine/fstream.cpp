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

//============================================================
// Input only FStream
//

/** File stream with output functions filtered out. */
size_t InputOnlyFStream::writeAvailable( int32 )
{
   throwUnsupported();
}

size_t InputOnlyFStream::write( const void*, size_t )
{
   throwUnsupported();
}

bool InputOnlyFStream::truncate(off_t pos)
{
   throwUnsupported();
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
}


size_t OutputOnlyFStream::read( void *, size_t size )
{
   throwUnsupported();
}


OutputOnlyFStream* OutputOnlyFStream::clone() const
{
   return new OutputOnlyFStream( *this );
}


//============================================================
// Read only FStream
//

off_t ReadOnlyFStream::seek( off_t pos, Stream::e_whence whence )
{
   throwUnsupported();
}


off_t ReadOnlyFStream::tell()
{
   throwUnsupported();
}


ReadOnlyFStream* ReadOnlyFStream::clone() const
{
   return new ReadOnlyFStream( *this );
}

//============================================================
// Write only FStream
//


off_t WriteOnlyFStream::seek( off_t pos, Stream::e_whence whence )
{
   throwUnsupported();
}

off_t WriteOnlyFStream::tell()
{
   throwUnsupported();
}

bool WriteOnlyFStream::truncate(off_t pos )
{
   throwUnsupported();
}


WriteOnlyFStream* WriteOnlyFStream::clone() const
{
   return new WriteOnlyFStream(*this);
}

}

/* end of fstream.cpp */
