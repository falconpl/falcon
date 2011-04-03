/*
   FALCON - The Falcon Programming Language.
   FILE: stmtrule.h

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTRULE_H
#define FALCON_STMTRULE_H

#include <falcon/statement.h>
#include <falcon/rulesyntree.h>

namespace Falcon
{

/** Rule statement.

   The rule statement processes one or more sub-trees in a rule context.
*/
class FALCON_DYN_CLASS StmtRule: public Statement
{
public:
   StmtRule( int32 line=0, int32 chr=0 );
   virtual ~StmtRule();
   
   StmtRule& addStatement( Statement* stmt );
   StmtRule& addAlternative();

   void describe( String& tgt ) const;
   inline String describe() const { return PStep::describe(); }

   static void apply_( const PStep*, VMachine* vm );

protected:
   typedef std::vector<RuleSynTree> AltTrees;
   AltTrees m_altTrees;
};

/** Cut statement.
   Kills the current rule context in a rule.
*/
class FALCON_DYN_CLASS StmtCut: public Statement
{
public:
   StmtCut( int32 line=0, int32 chr=0 );
   virtual ~StmtCut();

   void describe( String& tgt ) const;
   inline String describe() const { return PStep::describe(); }

   static void apply_( const PStep*, VMachine* vm );
};

}

#endif

/* end of stmtrule.h */
