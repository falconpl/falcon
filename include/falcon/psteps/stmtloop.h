/*
   FALCON - The Falcon Programming Language.
   FILE: stmtloop.h

   Statatement -- loop
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 06 Feb 2013 12:49:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STATLOOP_H
#define FALCON_STATLOOP_H

#include <falcon/psteps/stmtwhile.h>

namespace Falcon
{

/** Loop statement.
 *
 * Loops in a set of statements (syntree) indefinitely, or
 * possibly until an end condition comes true.
 */
class FALCON_DYN_CLASS StmtLoop: public StmtWhile
{
public:
   /** Creates a while statement for deserialization. */   
   StmtLoop( int32 line=0, int32 chr=0 );
   
   /** Creates a while statement, creating a new syntree. */   
   StmtLoop( Expression* expr, int32 line=0, int32 chr = 0 );
   
   /** Creates a while statement, adopting an existign statement set. */
   StmtLoop( TreeStep* stmts, int32 line=0, int32 chr = 0 );

   /** Creates a while statement, adopting an existign statement set. */
   StmtLoop( Expression* expr, TreeStep* stmts, int32 line=0, int32 chr = 0 );
   
   StmtLoop( const StmtLoop& other );
   
   virtual ~StmtLoop();

   virtual void describeTo( String& tgt, int depth=0 ) const;
   virtual void oneLinerTo( String& tgt ) const;
   virtual StmtLoop* clone() const { return new StmtLoop(*this); }

   virtual bool selector( Expression* e );

   static void apply_withexpr_( const PStep*, VMContext* ctx );
   static void apply_pure_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtwhile.h */
