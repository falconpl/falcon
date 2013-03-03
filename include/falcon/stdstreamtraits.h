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

#ifndef _FALCON_STDSTREAMTRAITS_H_
#define _FALCON_STDSTREAMTRAITS_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/streamtraits.h>

namespace Falcon
{

/**
 * Traits for the streams declared in the engine.
 */
class FALCON_DYN_CLASS StdStreamTraits
{
public:
   StdStreamTraits();
   ~StdStreamTraits();

   StreamTraits* stringStreamTraits() const { return m_stringStreamTraits; }
   StreamTraits* diskFileTraits() const { return m_diskFileTraits; }
   StreamTraits* readPipeTraits() const { return m_readPipeTraits; }
   StreamTraits* writePipeTraits() const { return m_writePipeTraits; }

private:
   StreamTraits* m_stringStreamTraits;
   StreamTraits* m_diskFileTraits;
   StreamTraits* m_readPipeTraits;
   StreamTraits* m_writePipeTraits;
};

}

#endif

/* end of stdstreamtraits.h */
