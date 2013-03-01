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

namespace Falcon
{

StdStreamTraits::StdStreamTraits()
{
   m_stringStreamTraits = new StringStream::Traits;
   m_diskFileTraits = 0;
   m_readPipeTraits = 0;
   m_writePipeTraits = 0;
}

StdStreamTraits::~StdStreamTraits()
{
   delete m_stringStreamTraits;
   delete m_diskFileTraits;
   delete m_readPipeTraits;
   delete m_writePipeTraits;
}

}

#endif

/* end of stdstreamtraits.g */
