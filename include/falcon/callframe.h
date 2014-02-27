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
class ClosedData;
class PStep;

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
   
   /** Data closed in closures. */
   ClosedData* m_closure;

   /** Data closed in closures. */
   ClosedData* m_closingData;

   /** The step calling this function. */
   const PStep* m_caller;

   /** Number of parameters used for the effective call. */
   uint32 m_paramCount;

   /** Stack base for this frame; item at this point is parameter 0 */
   uint32 m_dataBase;

   /** Local symbols stack base.
    \TODO This might be a temporary solution. Needs to be tested for performance
    and alternatives.
    */
   uint32 m_locsBase;

   /** Dynamic symbols stack base.
    \TODO This might be a temporary solution. Needs to be tested for performance
    and alternatives.
    */
   uint32 m_dynsBase;

   uint32 m_dynDataBase;

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

   /** True if self has been passed. */
   bool m_bMethodic;

   // Actually never used, just used at compile time by vector.
   CallFrame()
   {}

   CallFrame( Function* f, uint32 pc, uint32 sb, uint32 cb, uint32 dynb, uint32 locb, uint32 dynDataBase, const Item& self ):
      m_function(f),
      m_closure(0),
      m_closingData(0),
      m_paramCount( pc ),
      m_dataBase( sb ),
      m_locsBase( locb ),
      m_dynsBase( dynb ),
      m_dynDataBase(dynDataBase),
      m_codeBase( cb ),
      m_self(self),
      m_bMethodic( true )
   {}

   CallFrame( Function* f, uint32 pc, uint32 sb, uint32 cb, uint32 dynb, uint32 locb, uint32 dynDataBase ):
      m_function(f),
      m_closure(0),
      m_closingData(0),
      m_paramCount( pc ),
      m_dataBase( sb ),
      m_locsBase( locb ),
      m_dynsBase( dynb ),
      m_dynDataBase(dynDataBase),
      m_codeBase( cb ),
      m_bMethodic( false )
   {}
};

}

#endif

/* end of callframe.h */
