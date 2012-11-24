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

#define SRC "engine/psteps/breakpoint.cpp"
#include <falcon/psteps/pstep_compile.h>

namespace Falcon {

PStepCompile::PStepCompile(int32 line, int32 chr):
         PStep( line, chr )
{
   // This is not a syntactic pstep, we don't need to have a syntax for this.
   apply = apply_;
   m_compiler = new IntCompiler;
}

PStepCompile::PStepCompile( const PStepCompile& other ):
         PStep( other )
{
   apply = apply_;
   m_compiler = new IntCompiler;
}

PStepCompile::~PStepCompile()
{
   delete m_compiler;
}

PStepCompile* PStepCompile::clone() const
{
   return new PStepCompile(*this);
}

void PStepCompile::describeTo( String& tgt, int depth=0 ) const
{
   tgt ="<compile>";
}

static void PStepCompile::apply_( const PStep* ps, VMContext* ctx )
{
   PStepCompile* psc = static_cast<PStepCompile*>(ps);
   IntCompiler* comp = psc->m_compiler;
   comp->setCallingContext(ctx);

   comp->compileNext()
}

}

/* end of pstep_compile.h */
