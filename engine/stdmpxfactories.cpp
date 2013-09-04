/*
   FALCON - The Falcon Programming Language.
   FILE: stdmpxfactories.h

   Multiplex factories for the streams declared in the engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 28 Feb 2013 18:18:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/stdmpxfactories.cpp"

#include <falcon/stdmpxfactories.h>
#include <falcon/stringstream.h>
#include <falcon/diskmpxfactory.h>
#include <falcon/pipe.h>

namespace Falcon
{

StdMpxFactories::StdMpxFactories()
{
   m_stringStreamTraits = new StringStream::MpxFactory;
   m_diskFileTraits = new DiskMpxFactory;
   m_readPipeTraits = new Sys::Pipe::MpxFactory(true);
   m_writePipeTraits = new Sys::Pipe::MpxFactory(false);
}

StdMpxFactories::~StdMpxFactories()
{
   delete m_stringStreamTraits;
   delete m_diskFileTraits;
   delete m_readPipeTraits;
   delete m_writePipeTraits;
}

}

/* end of stdmpxfactories.cpp */
