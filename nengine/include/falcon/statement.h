/*
   FALCON - The Falcon Programming Language.
   FILE: statement.h

   Syntactic tree item definitions -- statements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STATEMENT_H
#define FALCON_STATEMENT_H

#include <falcon/pstep.h>
#include <falcon/pcode.h>
#include <falcon/vm.h>

namespace Falcon
{

class Expression;
class SynTree;

/** Statement.
 * Statements are PStep that may require other sub-sequences to be evaluated.
 * In other words, they are
 */
class FALCON_DYN_CLASS Statement: public PStep
{

public:
   typedef enum {
      breakpoint_t,
      autoexpr_t,
      if_t,
      while_t,
      return_t,
      rule_t,
      cut_t,
      init_t,
      custom_t
   } statement_t ;

   Statement( statement_t type, int32 line=0, int32 chr=0 ):
      PStep( line, chr ),
      m_step0(0), m_step1(0), m_step2(0), m_step3(0),
      m_discardable(false),
      m_type(type)
   {}

   inline virtual ~Statement() {}
   inline statement_t type() const { return m_type; }
   /** Subclasses can set this to true to be discareded during parsing.*/
   inline bool discardable() const { return m_discardable; }

protected:
   /** Steps being prepared by the statement */
   PStep* m_step0;
   PStep* m_step1;
   PStep* m_step2;
   PStep* m_step3;

   bool m_discardable;
   
   inline void prepare( VMContext* ctx ) const
   {
      if ( m_step0 )
      {
         ctx->pushCode(m_step0);
         if ( m_step1 )
         {
            ctx->pushCode(m_step1);
            if ( m_step2 )
            {
               ctx->pushCode(m_step2);
               if ( m_step3 )
               {
                  ctx->pushCode(m_step3);
               }
            }
         }
      }
   }

   friend class SynTree;
   friend class RuleSynTree;
private:
   statement_t m_type;
};

/** Statement causing the VM to return.

 This is a debug feature that causes the VM to return from its main
 loop when it meets this statement.
 */
class FALCON_DYN_CLASS Breakpoint: public Statement
{
public:
   Breakpoint(int32 line=0, int32 chr = 0);
   virtual ~Breakpoint();

   void describeTo( String& tgt ) const;

   static void apply_( const PStep*, VMContext* ctx );
};


/** Return statement.
 *
 * Exits the current function.
 */
class FALCON_DYN_CLASS StmtReturn: public Statement
{
public:
   /** Returns a value */
   StmtReturn( Expression* expr = 0, int32 line=0, int32 chr = 0 );
   virtual ~StmtReturn();

   void describeTo( String& tgt ) const;
   static void apply_( const PStep*, VMContext* ctx );

   Expression* expression() const { return m_expr; }
   void expression( Expression* expr );
private:
   Expression* m_expr;
   PCode m_pcExpr;
};


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
   PCode m_pcCheck;
   SynTree* m_stmts;
};


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

/* end of statement.h */
