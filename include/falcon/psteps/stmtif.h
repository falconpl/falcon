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
   StmtIf( SynTree* ifTrue, SynTree* ifFalse, int32 line=0, int32 chr = 0 );
   StmtIf( SynTree* ifTrue, int32 line=0, int32 chr = 0 );
   StmtIf( int32 line=0, int32 chr = 0 );
   StmtIf( const StmtIf& other );
   virtual ~StmtIf();

   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;
   virtual StmtIf* clone() const { return new StmtIf(*this); }
   
   static void apply_( const PStep*, VMContext* ctx );

   /** Adds an else-if branch to the if statement.
    If the branch has no selector, it becomes the else branch.
    */
   StmtIf& addElif( SynTree* ifTrue );

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;   
   virtual bool setNth( int32 n, TreeStep* ts );   
   virtual bool insert( int32 pos, TreeStep* element );   
   virtual bool remove( int32 pos );
   
private:
   class Private;
   Private* _p;
};

}

#endif

/* end of stmtif.h */
