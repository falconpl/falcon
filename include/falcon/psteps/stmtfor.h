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
   virtual void render( TextWriter* , int32  ) const {};
};


class FALCON_DYN_CLASS StmtForBase: public Statement
{
public:
   TreeStep* body() const { return m_body; }
   void body( TreeStep* st );

   TreeStep* forFirst() const { return m_forFirst; }
   void forFirst( TreeStep* st );

   TreeStep* forMiddle() const { return m_forMiddle; }
   void forMiddle( TreeStep* st );

   TreeStep* forLast() const { return m_forLast; }
   void forLast( TreeStep* st );

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );

   virtual bool isValid() const = 0;
   virtual bool isForHost() const { return true; }

   void minimize();

   virtual bool setTargetFromParam(Item* param) = 0;
   virtual bool setSelectorFromParam(Item* param) = 0;
   bool setBodyFromParam(Item* param);
   bool setForFirstFromParam(Item* param);
   bool setForMiddleFromParam(Item* param);
   bool setForLastFromParam(Item* param);

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual void renderHeading( TextWriter* tw, int32 depth ) const = 0;
protected:

   TreeStep* m_body;
   TreeStep* m_forFirst;
   TreeStep* m_forMiddle;
   TreeStep* m_forLast;

   StmtForBase( int32 line=0, int32 chr=0 ):
      Statement( line, chr ),
      m_body(0),
      m_forFirst(0),
      m_forMiddle(0),
      m_forLast(0)
      {}

   StmtForBase( const StmtForBase& other );

   virtual ~StmtForBase();

   class FALCON_DYN_CLASS PStepCleanup: public PStep
   {
   public:
      PStepCleanup() {
         apply = apply_;
         m_bIsLoopBase = true;
         // act also as a next-base when the loop is over.
         m_bIsNextBase = true;
      }
      virtual ~PStepCleanup() {};
      virtual void describeTo( String& str ) const { str = "PStepCleanup"; }

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

   /** Returns the generator associated with this for/in statement. */
   TreeStep* generator() const { return m_expr; }

   /** Adds an item expansion parameter. */
   void addParameter( const Symbol* sym );

   /** Arity of the for/in targets. */
   length_t paramCount() const;

   /** Gets the nth parameter. */
   const Symbol* param( length_t p ) const;

   void expandItem( Item& itm, VMContext* ctx ) const;

   virtual TreeStep* selector() const;
   virtual bool selector( TreeStep* expr );
   virtual StmtForIn* clone() const { return new StmtForIn(*this); }

   virtual bool isValid() const;

   virtual bool setTargetFromParam(Item* param);
   virtual bool setSelectorFromParam(Item* param);
   virtual void renderHeading( TextWriter* tw, int32 depth ) const;

private:
   class Private;
   Private* _p;

   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );

   class FALCON_DYN_CLASS PStepBegin: public PStep {
   public:
      PStepBegin( StmtForIn* owner ): m_owner(owner) { m_bIsLoopBase = true; apply = apply_; }
      virtual ~PStepBegin() {};
      virtual void describeTo( String& str ) const { str = "PStepBegin"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };

   class FALCON_DYN_CLASS PStepFirst: public PStep {
   public:
      PStepFirst( StmtForIn* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepFirst() {};
      virtual void describeTo( String& str ) const { str = "PStepFirst"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };

   class FALCON_DYN_CLASS PStepGetFirst: public PStep {
   public:
      PStepGetFirst( StmtForIn* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepGetFirst() {};
      virtual void describeTo( String& str ) const { str = "PStepGetFirst"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };

   class FALCON_DYN_CLASS PStepNext: public PStep {
   public:
      PStepNext( StmtForIn* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PStepNext() {};
      virtual void describeTo( String& str ) const { str = "PStepNext"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForIn* m_owner;
   };

   class FALCON_DYN_CLASS PStepGetNext: public PStep {
   public:
      PStepGetNext( StmtForIn* owner ): m_owner(owner) { m_bIsNextBase = true; apply = apply_; }
      virtual ~PStepGetNext() {};
      virtual void describeTo( String& str ) const { str = "PStepGetNext"; }

      //Need to do something about this
      StmtForIn* m_owner;
   private:
      static void apply_( const PStep* self, VMContext* ctx );

   };


   TreeStep* m_expr;

   PStepBegin m_stepBegin;
   PStepFirst m_stepFirst;
   PStepGetFirst m_stepGetFirst;
   PStepNext m_stepNext;
   PStepGetNext m_stepGetNext;

   friend class PStepGetFirst;
};

/** For/to statement.

 */
class FALCON_DYN_CLASS StmtForTo: public StmtForBase
{
public:
   StmtForTo( const Symbol* tgt=0, Expression* start=0, Expression* end=0, Expression* step=0, int32 line=0, int32 chr = 0 );
   StmtForTo( const StmtForTo& other );
   virtual ~StmtForTo();

   const Symbol* target() const { return m_target; }
   void target( const Symbol* t ) { m_target = t; }

   Expression* startExpr() const { return m_start; }
   void startExpr( Expression* s );

   Expression* endExpr() const { return m_end; }
   void endExpr( Expression* s );

   Expression* stepExpr() const { return m_step; }
   void stepExpr( Expression* s );

   virtual StmtForTo* clone() const { return new StmtForTo(*this); }

   virtual bool isValid() const;

   virtual bool setTargetFromParam(Item* param);
   virtual bool setSelectorFromParam(Item* param);
   virtual bool setStartExprFromParam(Item* param);
   virtual bool setEndExprFromParam(Item* param);
   virtual bool setStepExprFromParam(Item* param);

   virtual void renderHeading( TextWriter* tw, int32 depth ) const;

private:
   // apply is the same as PCODE, but it also checks ND requests.
   static void apply_( const PStep* self, VMContext* ctx );

   const Symbol* m_target;

   Expression* m_start;
   Expression* m_end;
   Expression* m_step;

   class FALCON_DYN_CLASS PStepNext: public PStep {
   public:
      PStepNext( StmtForTo* owner ): m_owner(owner) {
         m_bIsNextBase = true;
         apply = apply_;
      }
      virtual ~PStepNext() {};
      virtual void describeTo( String& str ) const { str = "PStepNext"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      StmtForTo* m_owner;
   };
   PStepNext m_stepNext;
};

}

#endif

/* end of stmtfor.h */
