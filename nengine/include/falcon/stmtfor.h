/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfor.h

   Syntactic tree item definitions -- Autoexpression.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Aug 2011 17:28:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STMTFOR_H_
#define _FALCON_STMTFOR_H_

#include <falcon/statement.h>

namespace Falcon {

// temporary statement used to keep track of the forming FOR expression
class StmtTempFor: public Statement
{
public:
   StmtTempFor():
      Statement( custom_t )
   {
      // don't record us, we're temp.
      m_discardable = true;
   }

   ~StmtTempFor()
   {
   }
};


class FALCON_DYN_CLASS StmtForBase: public Statement
{
public:   
   virtual void describeTo( String& tgt ) const;
   
   SynTree* body() const { return m_body; }
   void body( SynTree* st ) { m_body = st; }
   
   SynTree* forFirst() const { return m_forFirst; }
   void forFirst( SynTree* st ) { m_forFirst = st; }

   SynTree* forMiddle() const { return m_forMiddle; }
   void forMiddle( SynTree* st ) { m_forMiddle = st; }

   SynTree* forLast() const { return m_forLast; }
   void forLast( SynTree* st ) { m_forLast = st; }

protected:
   
   SynTree* m_body;
   SynTree* m_forFirst;
   SynTree* m_forMiddle;
   SynTree* m_forLast;
   
   StmtForBase( Statement::t_statement t, int32 line=0, int32 chr = 0 ):
      Statement( t, line, chr ),
      m_body(0),
      m_forFirst(0),
      m_forMiddle(0),
      m_forLast(0)
      {}
      
   virtual ~StmtForBase();
};


/** For/in statement.
 
 */
class FALCON_DYN_CLASS StmtForIn: public StmtForBase
{
public:
   StmtForIn( Expression* gen, int32 line=0, int32 chr = 0 );
   
   virtual ~StmtForIn();
   
   void oneLinerTo( String& tgt ) const;

   /** Returns the generator associated with this for/in statement. */
   Expression* generator() const { return m_expr; }

   /** Adds an item expansion parameter. */
   void addParameter( Symbol* sym );
   
   /** Arity of the for/in targets. */
   length_t paramCount() const;
   
   /** Gets the nth parameter. */
   Symbol* param( length_t p ) const;
   
   void expandItem( Item& itm, VMContext* ctx ) const;

   
private:
   class Private;
   Private* _p;
      
   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );   
   
   class PStepFirst: public PStep {
   public:
      PStepFirst( StmtForIn* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepFirst() {};
      void describeTo( String& str ) { str = "PStepFirst of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };
   
   class PStepNext: public PStep {
   public:
      PStepNext( StmtForIn* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepNext() {};
      void describeTo( String& str ) { str = "PStepNext of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };
   
   class PStepGetNext: public PStep {
   public:
      PStepGetNext( StmtForIn* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepGetNext() {};
      void describeTo( String& str ) { str = "PStepGetNext of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };
 

   Expression* m_expr;
   PCode m_pcExpr;  
   
   PStepFirst m_stepFirst;
   PStepNext m_stepNext;
   PStepGetNext m_stepGetNext;
};

/** For/to statement.
 
 */
class FALCON_DYN_CLASS StmtForTo: public StmtForBase
{
public:
   StmtForTo( Symbol* tgt=0, int64 start=0, int64 end=0, int64 step=0, int32 line=0, int32 chr = 0 );      
   virtual ~StmtForTo();
      
   Expression* startExpr() const { return m_start; }
   void startExpr( Expression* s );
   
   Expression* endExpr() const { return m_end; }
   void endExpr( Expression* s );

   Expression* stepExpr() const { return m_step; }
   void stepExpr( Expression* s );

   int64 startInt() const { return m_istart; }
   void startInt( int64 v ) { m_istart = v; }
   int64 endInt() const { return m_iend; }
   void endInt( int64 v ) { m_iend = v; }
   int64 endStep() const { return m_istep; }
   void endStep( int64 v ) { m_istep = v; }
      
   void oneLinerTo( String& tgt ) const;
   
private:
   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );
   
   Symbol* m_target;
   
   Expression* m_start;
   PCode m_pcExprStart;

   Expression* m_end;
   PCode m_pcExprEnd;
   
   Expression* m_step;
   PCode m_pcExprStep;

   int64 m_istart;
   int64 m_iend;
   int64 m_istep;
   
   class PStepNext: public PStep {
   public:
      PStepNext( StmtForTo* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepNext() {};
      void describeTo( String& str ) { str = "PStepNext of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForTo* m_owner;
   };
   PStepNext m_stepNext;
   
   class PStepPushStart: public PStep {
   public:
      PStepPushStart( StmtForTo* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepPushStart() {};
      void describeTo( String& str ) { str = "PStepPushStart of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForTo* m_owner;
   };
   PStepPushStart m_stepPushStart;
   
   class PStepPushEnd: public PStep {
   public:
      PStepPushEnd( StmtForTo* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepPushEnd() {};
      void describeTo( String& str ) { str = "PStepPushEnd of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForTo* m_owner;
   };
   PStepPushEnd m_stepPushEnd;
   
   class PStepPushStep: public PStep {
   public:
      PStepPushStep( StmtForTo* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepPushStep() {};
      void describeTo( String& str ) { str = "PStepPushStep of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForTo* m_owner;
   };
   PStepPushStep m_stepPushStep;
       
};

}

#endif

/* end of stmtfor.h */
