/*
   FALCON - The Falcon Programming Language.
   FILE: stdstreamtraits.h

   Traits for the streams declared in the engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 18:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/stdstreamtraits.cpp"

#include <falcon/stdstreamtraits.h>
#include <falcon/stringstream.h>
#include <falcon/diskfiletraits.h>
#include <falcon/pipe.h>

namespace Falcon
{

StdStreamTraits::StdStreamTraits()
{
   m_stringStreamTraits = new StringStream::Traits;
   m_diskFileTraits = new DiskFileTraits;
   m_readPipeTraits = new Sys::Pipe::Traits(true);
   m_writePipeTraits = new Sys::Pipe::Traits(false);
}

StdStreamTraits::~StdStreamTraits()
{
   delete m_stringStreamTraits;
   delete m_diskFileTraits;
   delete m_readPipeTraits;
   delete m_writePipeTraits;
}

}

/* end of stdstreamtraits.cpp */
