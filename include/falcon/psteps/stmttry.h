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
   StmtTry( SynTree* body, int32 line=0, int32 chr = 0 );
   StmtTry( int32 line=0, int32 chr = 0 );
   virtual ~StmtTry();

   virtual void describeTo( String& tgt ) const;
   void oneLinerTo( String& tgt ) const;

   /** Gets the body of this try. */
   SynTree* body() const { return m_body; }
   
   /** Gets the body for the finally block */   
   SynTree* fbody() const { return m_fbody; }
   /** Sets the body for the finally block.
    \param b The new body.
    
    Be sure that the statement didn't have a finally body already
    before invoking this method..
    */   
   void fbody(SynTree* b);
   
   StmtSelect& catchSelect() { return m_select; } 
   const StmtSelect& catchSelect() const { return m_select; }

private:
   SynTree *m_body;
   SynTree *m_fbody;
   StmtSelect m_select;
   
   static void apply_( const PStep*, VMContext* ctx );
   
   /** Placeholder for break after having pushed our body.*/
   class PStepDone: public PStep 
   {
   public:
      PStepDone() { apply = apply_; m_bIsCatch = true; }
      virtual ~PStepDone() {}
      virtual void describeTo( String& tgt ) const { tgt = "Try done"; }
   
   private:
      static void apply_( const PStep*, VMContext* ctx );
   }
   m_stepDone;   
   
   /** Execute the finally clause of this try.*/
   class PStepFinally: public PStep 
   {
   public:
      PStepFinally( StmtTry* t ): m_owner(t) { apply = apply_; m_bIsFinally = true; }
      virtual ~PStepFinally() {}
      virtual void describeTo( String& tgt ) const { tgt = "Try finally"; }
   
   private:
      StmtTry* m_owner;
      static void apply_( const PStep*, VMContext* ctx );
   }
   m_finallyStep;

   /** Cleanup step, popping the try frame and eventually invoking the finally.*/
   class PStepCleanup: public PStep 
   {
   public:
      PStepCleanup() { apply = apply_; }
      virtual ~PStepCleanup() {}
      virtual void describeTo( String& tgt ) const { tgt = "Try cleanup"; }
   
   private:
      static void apply_( const PStep*, VMContext* ctx );
   }
   m_cleanStep;
};

}

#endif

/* end of stmttry.h */
