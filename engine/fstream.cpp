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
