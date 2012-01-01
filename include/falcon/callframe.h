/*
   FALCON - The Falcon Programming Language.
   FILE: callframe.h

   Falcon virtual machine - call frame.
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
class ItemReference;


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
   
   /** Data closed in closures.. */
   ItemReference* m_closedData;

   /** Number of parameters used for the effective call. */
   uint32 m_paramCount;

   /** Stack base for this frame; item at this point is parameter 0 */
   uint32 m_stackBase;

   /** Stack base for this frame when the function is invoked.
    Rules can temporarily change the stack base. This stackbase is granted
    to be the initial stack base of the function.
    */
   uint32 m_initBase;

   /** Codebase for this frame.
    *
    * Code from this function is placed in this position; resizing the
    * codestack to this size activates the calling code.
    *
    * Actually, it's the codebase that directs the dance; this is used
    * only in case of explicit return so that it is not necessary to
    * scan the code stack to unroll the function call.
    */
   uint32 m_codeBase;
   
   /** Image of "self" in this frame. */
   Item m_self;

   /** Count of finally blocks traversed in this frame. */
   uint16 m_finallyCount;
   
   /** True if self has been passed. */
   bool m_bMethodic;
   
   

   // Actually never used, just used at compile time by vector.
   CallFrame()
   {}

   CallFrame( Function* f, int pc, int sb, int cb, const Item& self ):
      m_function(f),
      m_closedData(0),
      m_paramCount( pc ),
      m_stackBase( sb ),
      m_initBase( sb ),
      m_codeBase( cb ),
      m_self(self),
      m_finallyCount(0),
      m_bMethodic( true )
   {}

   CallFrame( Function* f, int pc, int sb, int cb):
      m_function(f),
      m_closedData(0),
      m_paramCount( pc ),
      m_stackBase( sb ),
      m_initBase( sb ),
      m_codeBase( cb ),
      m_finallyCount(0),
      m_bMethodic( false )
   {}
};

}

#endif

/* end of callframe.h */
