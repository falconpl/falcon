/*
   FALCON - The Falcon Programming Language.
   FILE: iff.h

   Falcon core module -- Functional if
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 13 Jan 2012 15:12:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_IFF_H
#define FALCON_CORE_IFF_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function iff
   @inset functional
   @param check The check that must be performed.
   @param iftrue Expression evaluated when check is true.
   @param iffalse Expression evaluated when check is false.
   @brief Perform a functional branching between expressions.

   @note This is an ETA function: the parameters are passed literally,
   as-is to the function, and are eventually evaluated during its execution.
*/

class FALCON_DYN_CLASS Iff: public Function
{
public:   
   Iff();
   virtual ~Iff();
   virtual void invoke( VMContext* ctx, int32 nParams );
   
public:
   
   class PStepChoice: public PStep
   {
   public:
      PStepChoice() { apply = apply_; }
      virtual ~PStepChoice() {};
      virtual void describeTo( String& str, int ) const { str = "PStepChoice"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
   };
   
   PStepChoice m_decide;
};

}
}

#endif	

/* end of iff.h */
