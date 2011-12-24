/*
   FALCON - The Falcon Programming Language.
   FILE: expression.h

   Syntactic tree item definitions -- expression elements.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRESSION_H
#define FALCON_EXPRESSION_H

#include <falcon/setup.h>
#include <falcon/pstep.h>
#include <falcon/sourceref.h>

namespace Falcon
{

class DataReader;
class DataWriter;
class PCode;
class PseudoFunction;
class Symbol;

/** Pure abstract class representing a Falcon expression.
 *
 * Base for all the expressions in the language.
 */
class FALCON_DYN_CLASS Expression: public PStep
{
public:
   typedef enum {
      t_value,
      t_symbol,
      t_neg,
      t_not,

      t_and,
      t_gate_and,
      t_or,
      t_gate_or,

      t_assign,

      t_plus,
      t_minus,
      t_times,
      t_divide,
      t_modulo,
      t_power,
      t_shr,
      t_shl,

      t_gt,
      t_ge,
      t_lt,
      t_le,
      t_eq,
      t_exeq,
      t_neq,

      /* We'll use this later */
      t_pre_inc,
      t_post_inc,
      t_pre_dec,
      t_post_dec,

      t_in,
      t_notin,
      t_provides,

      t_iif,

      t_obj_access,
      t_funcall,
      t_array_access,
      t_array_byte_access,
      t_strexpand,
      t_indirect,


      t_fbind,

      t_aadd,
      t_asub,
      t_amul,
      t_adiv,
      t_amod,
      t_apow,
      t_ashr,
      t_ashl,

      t_eval,
      t_oob,
      t_deoob,
      t_xoroob,
      t_isoob,

      t_arrayDecl,
      t_dictDecl,
      t_unpack,
      t_multiunpack,
      t_prototype,
      t_range,
         
      t_band,
      t_bor,
      t_bxor,
      t_bnot,

      t_self,
      t_reference
   } operator_t;

   Expression( const Expression &other );
   virtual ~Expression();

   /** Returns the type of this expression. */
   operator_t type() const { return m_operator; }
   /** Returns the position in the source where the expression was generated. */
   const SourceRef& sourceRef() const { return m_sourceRef; }
   /** Returns the position in the source where the expression was generated. */
   SourceRef& sourceRef() { return m_sourceRef; }

   /** Returns true if the expression can be found alone in a statement. */
   inline virtual bool isStandAlone() const { return false; }

   /** Returns true if the expression is composed of just constants.
    * When this method returns true, the expression can be simplified at compile time.
    */
   virtual bool isStatic() const = 0;

   /** Step that should be performed if this expression is lvalue.    
    @return A valid pstep if l-value is possible, 0 if this expression has no l-vaue.
    
    The PStep of an expression generates a value. The l-value pstep will use this
    expression to set a value when an assignment is required.
    */
   inline PStep* lvalueStep() const { return m_pstep_lvalue; }

   /** Clone this expression.
    */
   virtual Expression* clone() const = 0;

   /** Evaluates the expression when all its components are static.
    * @Return true if the expression can be simplified, false if it's not statitc
    *
    * Used during compilation to simplify static expressions, that is,
    * reducing expressions at compile time.
    */
   virtual bool simplify( Item& result ) const = 0;

protected:

   Expression( operator_t t ):
      m_pstep_lvalue(0),
      m_operator( t )
   {}
      
   /** Apply-modify function.
    
    Expression accepting a modify operator (i.e. ++, += *= etc.)
    can declare this modify step that will be used by auto-expression
    to perform the required modufy.
    
    If left uninitialized (to 0), this step won't be performed. This is the
    case of read-only expressions, i.e, function calls. In this case, 
    expressions like "call() += n" are legit, but they will be interpreted as
    "call() + n" as there is noting to be l-valued in "call()".
    
    \note It's supposed that the subclass own this pstep and sets it via &pstep,
    so that destruction of the pstep happens with the child class.
    */
   PStep* m_pstep_lvalue;
   
private:
   operator_t m_operator;
   SourceRef m_sourceRef;
};


/** Pure abstract Base unary expression. */
class FALCON_DYN_CLASS UnaryExpression: public Expression
{
public:
   inline UnaryExpression( operator_t type, Expression* op1 ):
      Expression( type ),
      m_first( op1 )
   {}

   UnaryExpression( const UnaryExpression& other );

   virtual ~UnaryExpression();

   virtual bool isStatic() const;

   Expression *first() const { return m_first; }
   void first( Expression *f ) { delete m_first; m_first= f; }

protected:
   Expression* m_first;

   inline UnaryExpression( operator_t type ):
         Expression( type ),
         m_first(0)
      {}
};


/** Pure abstract Base binary expression. */
class FALCON_DYN_CLASS BinaryExpression: public Expression
{
public:
   inline BinaryExpression( operator_t type, Expression* op1, Expression* op2 ):
         Expression( type ),
         m_first( op1 ),
         m_second( op2 )
      {}

   BinaryExpression( const BinaryExpression& other );

   virtual ~BinaryExpression();

   Expression *first() const { return m_first; }
   void first( Expression *f ) { delete m_first; m_first= f; }

   Expression *second() const { return m_second; }
   void second( Expression *s ) { delete m_second; m_second = s; }

   virtual bool isStatic() const;

protected:
   Expression* m_first;
   Expression* m_second;

   inline BinaryExpression( operator_t type ):
         Expression( type ),
         m_first(0),
         m_second(0)
      {}
};


/** Pure abstract Base ternary expression. */
class FALCON_DYN_CLASS TernaryExpression: public Expression
{
public:
   inline TernaryExpression( operator_t type, Expression* op1, Expression* op2, Expression* op3 ):
      Expression( type ),
      m_first( op1 ),
      m_second( op2 ),
      m_third( op3 )
   {}

   TernaryExpression( const TernaryExpression& other );

   virtual ~TernaryExpression();
   virtual bool isStatic() const;

   Expression *first() const { return m_first; }
   void first( Expression *f ) { delete m_first; m_first= f; }

   Expression *second() const { return m_second; }
   void second( Expression *s ) { delete m_second; m_second = s; }

   Expression *third() const { return m_third; }
   void third( Expression *t ) { delete m_third; m_third = t; }

protected:
   Expression* m_first;
   Expression* m_second;
   Expression* m_third;

   inline TernaryExpression( operator_t type ):
      Expression( type ),
      m_first(0),
      m_second(0),
      m_third(0)
   {}

};

//==============================================================
// Many operators take this form:
//

#define FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( class_name, op ) \
   inline class_name( Expression* op1 ): UnaryExpression( op, op1 ) {apply = apply_;} \
   inline class_name(): UnaryExpression( op, 0 ) {apply = apply_; }\
   inline class_name( const class_name& other ): UnaryExpression( other ) {apply = apply_;} \
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual void describeTo( String& ) const;\

#define FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( class_name, op ) \
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, op, )

#define FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, op, extended_constructor ) \
   inline class_name(): BinaryExpression( op ) {apply = apply_; extended_constructor}\
   inline class_name( Expression* op1, Expression* op2 ): BinaryExpression( op, op1, op2 ) { apply = apply_; extended_constructor } \
   inline class_name( const class_name& other ): BinaryExpression( other ) {apply = apply_; extended_constructor} \
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual void describeTo( String& ) const;\
   public:

#define FALCON_TERNARY_EXPRESSION_CLASS_DECLARATOR( class_name, op ) \
   inline class_name( Expression* op1, Expression* op2, Expression* op3 ): TernaryExpression( op, op1, op2, op3 ) {apply = apply_;} \
   inline class_name( const class_name& other ): TernaryExpression( other ) {apply = apply_;} \
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual void describeTo( String& ) const;\
   protected:\
   inline class_name(): TernaryExpression( op ) {}\
   public:

//==============================================================

/** logic not. */
class FALCON_DYN_CLASS ExprNot: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprNot, t_not );
};

/** logic and. */
class FALCON_DYN_CLASS ExprAnd: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprAnd, t_and );

   /** Check if the and expression can stand alone.
    *
    * An "and" expression can stand alone if it has a standalone second operator.
    */
   inline virtual bool isStandAlone() const { return m_second->isStandAlone(); }

private:
   class FALCON_DYN_CLASS Gate: public PStep {
   public:
      Gate();
      static void apply_( const PStep*, VMContext* ctx );
      mutable int m_shortCircuitSeqId;
   } m_gate;
};


/** logic or. */
class FALCON_DYN_CLASS ExprOr: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprOr, t_or );

   /** Check if the and expression can stand alone.
    *
    * An "or" expression can stand alone if it has a standalone second operand.
    */
   inline virtual bool isStandAlone() const { return m_second->isStandAlone(); }
};

/** Assignment operation. */
class FALCON_DYN_CLASS ExprAssign: public BinaryExpression
{
public:
   inline ExprAssign( Expression* op1, Expression* op2 ):
      BinaryExpression( t_assign, op1, op2 )
   {
   }

   inline ExprAssign( const ExprAssign& other ):
      BinaryExpression( other )
   {
   }

   inline virtual ExprAssign* clone() const { return new ExprAssign( *this ); }

   virtual bool simplify( Item& value ) const;
   virtual void describeTo( String& ) const;

   inline virtual bool isStandAlone() const { return true; }

protected:
   inline ExprAssign():
      BinaryExpression( t_assign ) {}
      
   static void apply_( const PStep* ps, VMContext* ctx );
};



/** Unary negative. */
class FALCON_DYN_CLASS ExprNeg: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprNeg, t_neg );
};

class FALCON_DYN_CLASS ExprBNOT: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprBNOT, t_bnot );
};

/** Exactly equal to operator. */
class FALCON_DYN_CLASS ExprEEQ: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprEEQ, t_exeq );
};

/** Array expansion. */
class FALCON_DYN_CLASS ExprUnpack: public Expression
{
public:
   ExprUnpack( Expression* op1, bool bIsTop );
   ExprUnpack( const ExprUnpack& other );
   virtual ~ExprUnpack();

   inline virtual ExprUnpack* clone() const { return new ExprUnpack( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void describeTo( String& ) const;

   int targetCount() const;
   Symbol* getAssignand( int n ) const;
   ExprUnpack& addAssignand( Symbol* );

   inline virtual bool isStandAlone() const { return false; }

   virtual bool isStatic() const { return false; }
   bool isTop() const { return m_bIsTop; }

protected:
   ExprUnpack();
   Expression* m_expander;
   bool m_bIsTop;
   
private:
   class Private;
   Private* _p;

   static void apply_( const PStep*, VMContext* ctx );
};


class FALCON_DYN_CLASS ExprMultiUnpack: public Expression
{
public:
   ExprMultiUnpack( bool bIsTop );
   ExprMultiUnpack( const ExprMultiUnpack& other );
   virtual ~ExprMultiUnpack();

   inline virtual ExprMultiUnpack* clone() const { return new ExprMultiUnpack( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void describeTo( String& ) const;

   int targetCount() const;
   Symbol* getAssignand( int n ) const;
   Expression* getAssignee( int n ) const;
   ExprMultiUnpack& addAssignment( Symbol* tgt, Expression* src );

   inline virtual bool isStandAlone() const { return false; }
   virtual bool isStatic() const { return false; }

   bool isTop() const { return m_bIsTop; }
protected:
   ExprMultiUnpack();
   bool m_bIsTop;

private:
   class Private;
   Private* _p;

   static void apply_( const PStep*, VMContext* ctx );
};

#if 0

/** "In" collection operator. */
class FALCON_DYN_CLASS ExprIn: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprIn, t_in );
};

/** "notIn" collection operator. */
class FALCON_DYN_CLASS ExprNotIn: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprNotIn, t_notin );
};

/** "provides" oop operator. */
class FALCON_DYN_CLASS ExprProvides: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprProvides, t_provides );
};

#endif

/** Fast if -- ternary conditional operator. */
class FALCON_DYN_CLASS ExprIIF: public TernaryExpression
{
public:
   inline ExprIIF( Expression* op1, Expression* op2, Expression* op3 ):
         TernaryExpression( t_iif, op1, op2, op3 ),
         m_gate( this )
         {apply = apply_;}
   inline ExprIIF( const ExprIIF& other ): TernaryExpression( other ), m_gate(this) {apply = apply_;}
   inline virtual ExprIIF* clone() const { return new ExprIIF( *this ); }
   virtual bool simplify( Item& value ) const;
   static void apply_( const PStep*, VMContext* ctx );
   virtual void describeTo( String& ) const;
   
   /** Check if the and expression can stand alone.
      An "?" expression can stand alone if the second AND third operand are standalone.
    */
   inline virtual bool isStandAlone() const {
      return m_second->isStandAlone() && m_third->isStandAlone();
   }
   
protected:
      
   inline ExprIIF(): TernaryExpression( t_iif ), m_gate(this) {}

private:
   mutable int m_falseSeqId;
   
   class FALCON_DYN_CLASS Gate: public PStep {
   public:
      Gate( ExprIIF* owner );
      void describeTo( String& target ) const { target = "Gate for expriif"; }
      static void apply_( const PStep*, VMContext* ctx );
   private:
      ExprIIF* m_owner;
   } m_gate;
};


/** Special string Index accessor. */
class FALCON_DYN_CLASS ExprStarIndex: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprStarIndex, t_array_byte_access );
};

#if 0

/** String expansion expression */
class FALCON_DYN_CLASS ExprStrExpand: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprStrExpand, t_strexpand );
};

/** Indirect expression -- probably to be removed */
class FALCON_DYN_CLASS ExprIndirect: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprIndirect, t_indirect );
};

/** Future bind operation -- to be reshaped or removed */
class FALCON_DYN_CLASS ExprFbind: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprFbind, t_fbind );
};


/** Fast-evaluate expression. */
class FALCON_DYN_CLASS ExprEval: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprEval, t_eval );
   inline virtual bool isStandAlone() const { return true; }
};

#endif


/** Set Out-of-band expression. */
class FALCON_DYN_CLASS ExprOob: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprOob, t_oob );
};

/** Reset Out-of-band expression. */
class FALCON_DYN_CLASS ExprDeoob: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprDeoob, t_deoob );
};

/** Invert Out-of-band expression. */
class FALCON_DYN_CLASS ExprXorOob: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprXorOob, t_xoroob );
};

/** Check if is Out-of-band expression. */
class FALCON_DYN_CLASS ExprIsOob: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprIsOob, t_isoob );
};

//======================================
// Self &c

/** Class implementing Self atom value.
 */
class FALCON_DYN_CLASS ExprSelf: public Expression
{
public:
   ExprSelf();
   ExprSelf( const ExprSelf &other );
   virtual ~ExprSelf();

   virtual bool isStatic() const;
   virtual ExprSelf* clone() const;
   virtual bool simplify( Item& result ) const;
   virtual void describeTo( String & str ) const;

private:
   static void apply_( const PStep* s1, VMContext* ctx );

};

}

#endif

/* end of expression.h */

