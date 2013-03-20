/*
   FALCON - The Falcon Programming Language.
   FILE: stmttry.h

   Syntactic tree item definitions -- Try/catch.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 15 Aug 2011 13:58:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STMTTRY_H_
#define _FALCON_STMTTRY_H_

#include <falcon/pstep.h>
#include <falcon/psteps/stmtselect.h>

namespace Falcon {

class RequiredClass;

/** Implementation of the try/catch/finally statement.
 
 \TODO More docs.
 */
class FALCON_DYN_CLASS StmtTry: public Statement
{
public:   
   StmtTry( int32 line=0, int32 chr = 0 );   
   StmtTry( SynTree* body, int32 line=0, int32 chr = 0 );   
   StmtTry( SynTree* body, SynTree* fbody, int32 line=0, int32 chr = 0 );   
   StmtTry( const StmtTry& other );   
   virtual ~StmtTry();

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual StmtTry* clone() const { return new StmtTry(*this); }

   /** Gets the body of this try. */
   SynTree* body() const { return m_body; }
   /** Sets the body for the main block.
    \param b The new body.
    
    Be sure that the statement didn't have a finally body already
    before invoking this method..
    */   
   bool body(SynTree* b);
   
   
   /** Gets the body for the finally block */   
   SynTree* fbody() const { return m_fbody; }
   /** Sets the body for the finally block.
    \param b The new body.
    
    Be sure that the statement didn't have a finally body already
    before invoking this method..
    */   
   bool fbody(SynTree* b);
   
   StmtSelect& catchSelect() { return *m_select; }
   const StmtSelect& catchSelect() const { return *m_select; }

   /**
    * Try has arity 3: body, catch clauses (a statement block) and finally body.
    */
   virtual int32 arity() const;
   /**
    * Try has arity 3: body, catch clauses (a statement block) and finally body.
    */
   virtual TreeStep* nth( int32 n ) const;
   /**
    * Try has arity 3; setting 1 (body) will fail.
    */
   virtual bool setNth( int32 n, TreeStep* ts );

private:
   SynTree *m_body;
   SynTree *m_fbody;
   StmtSelect* m_select;
   
   static void apply_( const PStep*, VMContext* ctx );

   /** Execute the finally clause of this try.*/
   class FALCON_DYN_CLASS PStepFinally: public PStep
   {
   public:
      PStepFinally( StmtTry* t ): m_owner(t) { apply = apply_;}
      virtual ~PStepFinally() {}
      virtual void describeTo( String& tgt ) const { tgt = "Try finally"; }

   private:
      StmtTry* m_owner;
      static void apply_( const PStep*, VMContext* ctx );
   }
   m_finallyStep;

};

}

#endif

/* end of stmttry.h */
