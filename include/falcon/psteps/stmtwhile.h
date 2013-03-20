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
   /** Creates a while statement for deserialization. */   
   StmtWhile( int32 line=0, int32 chr=0 );
   
   /** Creates a while statement, creating a new syntree. */   
   StmtWhile( Expression* expr, int32 line=0, int32 chr = 0 );
   
   /** Creates a while statement, adopting an existign statement set. */
   StmtWhile( Expression* expr, TreeStep* stmts, int32 line=0, int32 chr = 0 );
   
   StmtWhile( const StmtWhile& other );
   
   virtual ~StmtWhile();

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual StmtWhile* clone() const { return new StmtWhile(*this); }
   static void apply_( const PStep*, VMContext* ctx );

   TreeStep* mainBlock() const { return m_child; }
   void mainBlock(TreeStep* st);
   TreeStep* detachMainBlock();
   
   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 pos, TreeStep* element );
   virtual bool append( TreeStep* element );
   
   virtual Expression* selector() const;   
   virtual bool selector( Expression* e ); 

   /** We can remove a bit of stuff. */
   virtual void minimize();

protected:
   TreeStep* m_child;
   Expression* m_expr;

   // changes our single child into a syntree with two children.
   void singleToMultipleChild( TreeStep* element, bool last );
};

}

#endif

/* end of stmtwhile.h */
