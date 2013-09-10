/*
   FALCON - The Falcon Programming Language.
   FILE: stdmpxfactories.h

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
#include <falcon/multiplex.h>

namespace Falcon
{

/**
 * Traits for the streams declared in the engine.
 */
class FALCON_DYN_CLASS StdMpxFactories
{
public:
   StdMpxFactories();
   ~StdMpxFactories();

   const Multiplex::Factory* stringStreamMpxFact() const { return m_stringStreamTraits; }
   const Multiplex::Factory* diskFileMpxFact() const { return m_diskFileTraits; }
   const Multiplex::Factory* readPipeMpxFact() const { return m_readPipeTraits; }
   const Multiplex::Factory* writePipeMpxFact() const { return m_writePipeTraits; }
   const Multiplex::Factory* fileDataMpxFact() const { return m_fileDataTraits; }

private:
   const Multiplex::Factory* m_stringStreamTraits;
   const Multiplex::Factory* m_diskFileTraits;
   const Multiplex::Factory* m_readPipeTraits;
   const Multiplex::Factory* m_writePipeTraits;
   const Multiplex::Factory* m_fileDataTraits;
};

}

#endif

/* end of stdmpxfactories.h */
