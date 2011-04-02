/*
   FALCON - The Falcon Programming Language.
   FILE: callframe.h

   Falcon virtual machine - code frame.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CALLFRAME_H
#define FALCON_CALLFRAME_H

#include <falcon/setup.h>
#include <falcon/item.h>

namespace Falcon {

class Function;

/** Call Frame for the Falcon virtual machine.
 *
 * The Falcon Virtual Machine keeps call frames away
 * from the other stacks for faster reference and simpler
 * call unrolling.
 *
 * Function* and self item in this structure is explicitly
 * marked by the GC at VM scan.
 *
*/
class FALCON_DYN_CLASS CallFrame
{
public:
   /** Function calling this frame. */
   Function* m_function;

   /** Number of parameters used for the effective call. */
   int m_paramCount;

   /** Stack base for this frame; item at this point is parameter 0 */
   int m_stackBase;

   /** Codebase for this frame.
    *
    * Code from this function is placed in this position; resizing the
    * codestack to this size activates the calling code.
    *
    * Actually, it's the codebase that directs the dance; this is used
    * only in case of explicit return so that it is not necessary to
    * scan the code stack to unroll the function call.
    */
   int m_codeBase;

   /** Image of "self" in this frame. */
   Item m_self;

   // Actually never used, just used at compile time by vector.
   CallFrame()
   {}

   CallFrame( Function* f, int pc, int sb, int cb, const Item& self ):
      m_function(f),
      m_paramCount( pc ),
      m_stackBase( sb ),
      m_codeBase( cb ),
      m_self(self)
   {}
};

}

#endif

/* end of callframe.h */
