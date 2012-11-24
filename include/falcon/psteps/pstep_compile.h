/*
   FALCON - The Falcon Programming Language.
   FILE: pstep_compile.h

   A step continuously invoking and evaluating a compilation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 21 Nov 2012 14:20:53 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_PSTEP_COMPILE_H_
#define _FALCON_PSTEP_COMPILE_H_

#include <falcon/pstep.h>
#include <falcon/intcompiler.h>

namespace Falcon {

/** Non-grammar pstep performing a local compilation.
 *
 * This is a special pstep. It should be never be shared across
 * contexts, but just created and stored in the context where
 * it is used.
 *
 * The owner should destroy it when the computation is performed.
 *
 */
class PStepCompile: public PStep
{
   PStepCompile(int32 line=0, int32 chr = 0);
   PStepCompile( const PStepCompile& other );
   virtual ~PStepCompile();
   PStepCompile* clone() const;

   void describeTo( String& tgt, int depth=0 ) const;

   static void apply_( const PStep*, VMContext* ctx );
private:
   IntCompiler* m_compiler;
};

}
#endif /* PSTEP_COMPILE_H_ */

/* end of pstep_compile.h */
