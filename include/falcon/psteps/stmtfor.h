/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfor.h

   Syntactic tree item definitions -- For/in and For/to.
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

class Symbol;

// temporary statement used to keep track of the forming FOR expression
class StmtTempFor: public Statement
{
public:
   StmtTempFor():
      Statement(  )
   {
      // don't record us, we're temp.
      m_discardable = true;
   }

   ~StmtTempFor()
   {
   }
   
   virtual StmtTempFor* clone() const { return 0; }
};


class FALCON_DYN_CLASS StmtForBase: public Statement
{
public:   
   virtual void describeTo( String& tgt, int depth=0 ) const;
   
   SynTree* body() const { return m_body; }
   void body( SynTree* st ) { m_body = st; }
   
   SynTree* forFirst() const { return m_forFirst; }
   void forFirst( SynTree* st ) { m_forFirst = st; }

   SynTree* forMiddle() const { return m_forMiddle; }
   void forMiddle( SynTree* st ) { m_forMiddle = st; }

   SynTree* forLast() const { return m_forLast; }
   void forLast( SynTree* st ) { m_forLast = st; }
   
   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool nth( int32 n, TreeStep* ts );

   virtual bool isValid() const = 0;
   virtual bool isForHost() const { return true; }
   
protected:
   
   SynTree* m_body;
   SynTree* m_forFirst;
   SynTree* m_forMiddle;
   SynTree* m_forLast;
   
   StmtForBase( int32 line=0, int32 chr=0 ):
      Statement( line, chr ),
      m_body(0),
      m_forFirst(0),
      m_forMiddle(0),
      m_forLast(0)
      {}
   
   StmtForBase( const StmtForBase& other );
      
   virtual ~StmtForBase();
   
   class PStepCleanup: public PStep
   {
   public:
      PStepCleanup() { 
         apply = apply_; 
         m_bIsLoopBase = true;
         // act also as a next-base when the loop is over.
         m_bIsNextBase = true; 
      }
      virtual ~PStepCleanup() {};
      void describeTo( String& str ) { str = "PStepCleanup"; }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
   };
   PStepCleanup m_stepCleanup;
   
   
};


/** For/in statement.
 
 */
class FALCON_DYN_CLASS StmtForIn: public StmtForBase
{
public:
   StmtForIn( int32 line=0, int32 chr = 0 );
   StmtForIn( const StmtForIn& other );
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

   virtual Expression* selector() const; 
   virtual bool selector( Expression* e );
   virtual StmtForIn* clone() const { return new StmtForIn(*this); }
   
   virtual bool isValid() const;
private:
   class Private;
   Private* _p;
      
   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );   
    
   class PStepBegin: public PStep {
   public:
      PStepBegin( StmtForIn* owner ): m_owner(owner) { m_bIsLoopBase = true; apply = apply_; }
      virtual ~PStepBegin() {};
      void describeTo( String& str, int=0 ) { str = "PStepBegin of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };
   
   class PStepFirst: public PStep {
   public:
      PStepFirst( StmtForIn* owner ): m_owner(owner) { m_bIsLoopBase = true; apply = apply_; }
      virtual ~PStepFirst() {};
      void describeTo( String& str, int = 0 ) { str = "PStepFirst of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };
   
   class PStepNext: public PStep {
   public:
      PStepNext( StmtForIn* owner ): m_owner(owner) { m_bIsLoopBase = true; apply = apply_; }
      virtual ~PStepNext() {};
      void describeTo( String& str ) { str = "PStepNext of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };
   
   class PStepGetNext: public PStep {
   public:
      PStepGetNext( StmtForIn* owner ): m_owner(owner) { m_bIsNextBase = true; apply = apply_; }
      virtual ~PStepGetNext() {};
      void describeTo( String& str, int =0 ) { str = "PStepGetNext of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };
 

   Expression* m_expr;
   
   PStepBegin m_stepBegin;
   PStepFirst m_stepFirst;
   PStepNext m_stepNext;
   PStepGetNext m_stepGetNext;
};

/** For/to statement.
 
 */
class FALCON_DYN_CLASS StmtForTo: public StmtForBase
{
public:
   StmtForTo( Symbol* tgt=0, Expression* start=0, Expression* end=0, Expression* step=0, int32 line=0, int32 chr = 0 );      
   StmtForTo( const StmtForTo& other );
   virtual ~StmtForTo();
      
   Expression* startExpr() const { return m_start; }
   void startExpr( Expression* s );
   
   Expression* endExpr() const { return m_end; }
   void endExpr( Expression* s );

   Expression* stepExpr() const { return m_step; }
   void stepExpr( Expression* s );
      
   virtual void oneLinerTo( String& tgt ) const;
   virtual StmtForTo* clone() const { return new StmtForTo(*this); }
   
   virtual bool isValid() const;
private:
   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );
   
   Symbol* m_target;
   
   Expression* m_start;
   Expression* m_end;
   Expression* m_step;
   
   class PStepNext: public PStep {
   public:
      PStepNext( StmtForTo* owner ): m_owner(owner) { 
         m_bIsNextBase = true; 
         apply = apply_; 
      }
      virtual ~PStepNext() {};
      void describeTo( String& str, int = 0 ) { str = "PStepNext of " + m_owner->oneLiner(); }
      
   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForTo* m_owner;
   };
   PStepNext m_stepNext;
   
       
};

}

#endif

/* end of stmtfor.h */
