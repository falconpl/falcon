/*
   FALCON - The Falcon Programming Language.
   FILE: pcode.h

   Falcon virtual machine - pre-compiled code
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 17:54:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_PCODE_H
#define FALCON_PCODE_H

#include <falcon/setup.h>
#include <falcon/pstep.h>
#include <vector>

namespace Falcon {

class StmtAutoexpr;

/** Pre-Compiled code for the Falcon virtual machine.
 *
 * Precompiled code is a LIFO structure of pre-compiled code.
 * The steps in a PCode are to be considered like pushed as-is
 * in the virtual machine code stack, but instead of being
 * compiled (pushed) on the fly, they have been pre-compiled
 * for performance.
 *
 * The VM "applies" a pre compiled code by simply applying
 * all the steps in it last-to-first.
 *
 * Also, the CompExpr won't call its member's prepare;
 * that has been pre-called during the expression compilation
 * phase. Instead, it will call their apply, as if they were
 * called directly by the VM.
 *
 * PCode execution is vm-atomic. It's particularly adequate
 * to expression evaluation, as the apply of a PCode is considered
 * a single VM step.
 *
 * @note Remember that the PSteps in the PCode belong to the
 * respective SynTree or statement holder. PCode is just a pre-pushed
 * (linearized) set of static PSteps held elsewhere.
 *
 * \note Important: the calling code should make sure that the
 * expression is precompiled at most ONCE, or in other words, that
 * the PCode on which is precompiled is actually used just once
 * in the target program. In fact, gate expressions uses a private
 * member in their structure to determine the jump branch position,
 * and that member can be used just once.
 *
*/
class FALCON_DYN_CLASS PCode: public PStep
{
public:

   PCode();
   inline int size() const { return m_steps.size(); }

   /** Pushes a new step in the pcode. */
   inline void pushStep( const PStep* ps ) { m_steps.push_back( ps ); }

   virtual void describe( String& res ) const;
   virtual String describe() const {return PStep::describe(); }

   friend class StmtAutoexpr;

private:
   typedef std::vector<const PStep*> StepList;
   StepList m_steps;

   static void apply_( const PStep* ps, VMContext* ctx );
};

}

#endif

/* end of compexpr.h */
