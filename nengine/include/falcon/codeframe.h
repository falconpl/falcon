/*
   FALCON - The Falcon Programming Language.
   FILE: codeframe.h

   Falcon virtual machine - code frame.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CODEFRAME_H
#define FALCON_CODEFRAME_H

#include <falcon/setup.h>
#include <falcon/pstep.h>

namespace Falcon {

/** Step frame for the Falcon virtual machine.
 *
 * The Falcon virtual machine executes a set of PStep elements
 * stored in a code stack. Each single data in the code stack
 * is composed of the PStep and a few additional information
 * which is kept in this class.
 *
*/
class FALCON_DYN_CLASS CodeFrame: public BaseAlloc
{
public:
   /** The pstep to be executed now. */
   PStep* m_step;

   /** Sequence ID (internal step in the sequence)  */
   int m_seqId;

   inline CodeFrame( PStep* ps ):
         m_step(ps),
         m_seqId(0)
   {}

   inline CodeFrame( const CodeFrame& other ):
      m_step(other.m_step),
      m_seqId(other.m_seqId)
   {}

};

}

#endif

/* end of codeframe.h */
