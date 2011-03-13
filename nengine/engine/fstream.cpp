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
int32 InputOnlyFStream::writeAvailable( int32, Interrupt* )
{
   throwUnsupported();
}

int32 InputOnlyFStream::write( const void *buffer, int32 size )
{
   throwUnsupported();
}

bool InputOnlyFStream::truncate(int64 pos)
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


int32 OutputOnlyFStream::readAvailable( int32, Interrupt* )
{
   throwUnsupported();
}


int32 OutputOnlyFStream::read( const void *buffer, int32 size )
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

int64 ReadOnlyFStream::seek( int64 pos, Stream::e_whence whence )
{
   throwUnsupported();
}


int64 ReadOnlyFStream::tell()
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


int64 WriteOnlyFStream::seek( int64 pos, Stream::e_whence whence )
{
   throwUnsupported();
}

int64 WriteOnlyFStream::tell()
{
   throwUnsupported();
}

bool WriteOnlyFStream::truncate(int64 pos )
{
   throwUnsupported();
}


WriteOnlyFStream* WriteOnlyFStream::clone() const
{
   return new WriteOnlyFStream(*this);
}

}

/* end of fstream.cpp */
