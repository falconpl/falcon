/*
   FALCON - The Falcon Programming Language.
   FILE: rulesyntree.h

   Syntactic tree item definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_RULESYNTREE_H
#define FALCON_RULESYNTREE_H

#include <falcon/syntree.h>

namespace Falcon
{

class Statement;

/** Syntactic tree for rules.
 *
 * Evaluates a single list of rules in a rule context.
 */
class FALCON_DYN_CLASS RuleSynTree: public SynTree
{

public:
   RuleSynTree();
   virtual ~RuleSynTree();

private:
   static void apply_( const PStep* ps, VMContext* ctx );

   class PStepNext: public PStep {
   public:
      PStepNext( RuleSynTree* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepNext() {};
      void describeTo( String& str ) { str = "PStepNext of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      RuleSynTree* m_owner;
   };
   PStepNext m_stepNext;
       
};

}

#endif

/* end of rulesyntree.h */
