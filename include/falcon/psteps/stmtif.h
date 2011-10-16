/*
   FALCON - The Falcon Programming Language.
   FILE: stmtif.h

   Statatement -- if (branch)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTIF_H
#define FALCON_STMTIF_H

#include <falcon/statement.h>

namespace Falcon
{

/** If statement.
 *
 * Main logic branch control.
 */
class FALCON_DYN_CLASS StmtIf: public Statement
{
public:
   StmtIf( Expression* check, SynTree* ifTrue, SynTree* ifFalse = 0, int32 line=0, int32 chr = 0 );
   virtual ~StmtIf();

   virtual void describeTo( String& tgt ) const;
   void oneLinerTo( String& tgt ) const;

   static void apply_( const PStep*, VMContext* ctx );

   /** Adds an else-if branch to the if statement */
   StmtIf& addElif( Expression *check, SynTree* ifTrue, int32 line=0, int32 chr = 0 );

   /** Sets the else branch for this if statement. */
   StmtIf& setElse( SynTree* ifFalse );

private:
   SynTree* m_ifFalse;

   struct Private;
   Private* _p;

};

}

#endif

/* end of stmtif.h */
