/*
   FALCON - The Falcon Programming Language.
   FILE: stmtwhile.h

   Statatement -- while
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:51:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STATWHILE_H
#define FALCON_STATWHILE_H

#include <falcon/statement.h>

namespace Falcon
{

/** While statement.
 *
 * Loops in a set of statements (syntree) while the given expression evaluates as true.
 */
class FALCON_DYN_CLASS StmtWhile: public Statement
{
public:
   StmtWhile( Expression* check, SynTree* stmts, int32 line=0, int32 chr = 0 );
   virtual ~StmtWhile();

   void describeTo( String& tgt ) const;
   void oneLinerTo( String& tgt ) const;
   static void apply_( const PStep*, VMContext* ctx );

private:   
   Expression* m_check;
   SynTree* m_stmts;
};

}

#endif

/* end of stmtwhile.h */
