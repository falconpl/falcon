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

#include <vector>

namespace Falcon
{

class Stream;
class ExprFactory;
class PCode;

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

      t_eval,
      t_oob,
      t_deoob,
      t_xoroob,
      t_isoob

   } operator_t;

   Expression( const Expression &other );
   virtual ~Expression();

   /** Returns the type of this expression. */
   operator_t type() const { return m_operator; }
   /** Returns the position in the source where the expression was generated. */
   const SourceRef& sourceRef() const { return m_sourceRef; }
   /** Returns the position in the source where the expression was generated. */
   SourceRef& sourceRef() { return m_sourceRef; }

   /** Serialize the expression on a stream. */
   virtual void serialize( Stream* s ) const;

   /** Returns true if the expression can be found alone in a statement. */
   inline virtual bool isStandAlone() const { return false; }

   /** True for binary expressions */
   virtual bool isBinaryOperator() const = 0;

   /** Sets the given expression as a target of an assignment.
    *
    * The base class implementation throws an error (assignment to non-lvalue).
    * Only assignable expressions (symbols and accessors) can be set as lvalue.
    *
    * This is automatically done when adding the expression as first member of
    * an assignment.
    */
   inline virtual void setLValue() {
      //TODO Throw proper exception
   }

   /** Tells if this expression is subject to l-value.
    * @return true if this expression is the left part of an assignment, false otherwise.
    */
   inline virtual bool isLValue() const { return false; }

   /** Returns true if the expression is composed of just constants.
    * When this method returns true, the expression can be simplified at compile time.
    */
   virtual bool isStatic() const = 0;

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


   /** Pre-compiles the expression on a PCode.
    *
    * The vast majority of expressions in Falcon programs can be
    * precompiled to PCode stack stubs, and then sent into the VM
    * through their precompiled PCode form.
    *
    * \note Important: the calling code should make sure that the
    * expression is precompiled at most ONCE, or in other words, that
    * the PCode on which is precompiled is actually used just once
    * in the target program. In fact, gate expressions uses a private
    * member in their structure to determine the jump branch position,
    * and that member can be used just once.
    */
   virtual void precompile( PCode* pcd ) const;

protected:

   Expression( operator_t t ):
      m_operator( t )
   {}

   /** Deserialize the expression from a stream.
    * The expression type id must have been already read.
    */
   virtual void deserialize( Stream* s );

   friend class ExprFactory;

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

   virtual void serialize( Stream* s ) const;
   virtual bool isStatic() const;

   inline virtual bool isBinaryOperator() const { return false; }

   Expression *first() const { return m_first; }
   void first( Expression *f ) { delete m_first; m_first= f; }

   virtual void precompile( PCode* pcd ) const;

protected:
   Expression* m_first;

   inline UnaryExpression( operator_t type ):
         Expression( type ),
         m_first(0)
      {}

   virtual void deserialize( Stream* s );

   friend class ExprFactory;
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

   virtual void serialize( Stream* s ) const;
   inline virtual bool isBinaryOperator() const { return true; }

   Expression *first() const { return m_first; }
   void first( Expression *f ) { delete m_first; m_first= f; }

   Expression *second() const { return m_second; }
   void second( Expression *s ) { delete m_second; m_second = s; }

   virtual bool isStatic() const;

   virtual void precompile( PCode* pcd ) const;

protected:
   Expression* m_first;
   Expression* m_second;

   inline BinaryExpression( operator_t type ):
         Expression( type ),
         m_first(0),
         m_second(0)
      {}

   virtual void deserialize( Stream* s );

   friend class ExprFactory;
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
   virtual void serialize( Stream* s ) const;
   inline virtual bool isBinaryOperator() const { return false; }
   virtual bool isStatic() const;

   Expression *first() const { return m_first; }
   void first( Expression *f ) { delete m_first; m_first= f; }

   Expression *second() const { return m_second; }
   void second( Expression *s ) { delete m_second; m_second = s; }

   Expression *third() const { return m_third; }
   void third( Expression *t ) { delete m_third; m_third = t; }

   virtual void precompile( PCode* pcd ) const;

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

   virtual void deserialize( Stream* s );

   friend class ExprFactory;
};

//==============================================================
// Many operators take this form:
//

#define FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( class_name, op ) \
   inline class_name( Expression* op1 ): UnaryExpression( op, op1 ) {apply = apply_;} \
   inline class_name( const class_name& other ): UnaryExpression( other ) {apply = apply_;} \
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMachine* vm ); \
   virtual void toString( String& ) const;\
   protected:\
   inline class_name(): UnaryExpression( op ) {}\
   friend class ExprFactory;\
   public:

#define FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( class_name, op ) \
   inline class_name( Expression* op1, Expression* op2 ): BinaryExpression( op, op1, op2 ) { apply = apply_; } \
   inline class_name( const class_name& other ): BinaryExpression( other ) {apply = apply_;} \
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMachine* vm ); \
   virtual void toString( String& ) const;\
   protected:\
   inline class_name(): BinaryExpression( op ) {}\
   friend class ExprFactory;\
   public:

#define FALCON_TERNARY_EXPRESSION_CLASS_DECLARATOR( class_name, op ) \
   inline class_name( Expression* op1, Expression* op2, Expression* op3 ): TernaryExpression( op, op1, op2, op3 ) {apply = apply_;} \
   inline class_name( const class_name& other ): TernaryExpression( other ) {apply = apply_;} \
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMachine* vm ); \
   virtual void toString( String& ) const;\
   protected:\
   inline class_name(): TernaryExpression( op ) {}\
   friend class ExprFactory;\
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

   void precompile( PCode* pcode ) const;

private:
   class FALCON_DYN_CLASS Gate: public PStep {
   public:
      Gate();
      static void apply_( const PStep*, VMachine* vm );
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
    * An "or" expression can stand alone if it has a standalone second operator.
    */
   inline virtual bool isStandAlone() const { return m_second->isStandAlone(); }

   virtual void precompile( PCode* pcode ) const;

private:

   class FALCON_DYN_CLASS Gate: public PStep {
   public:
      Gate();
      static void apply_( const PStep*, VMachine* vm );
      mutable int m_shortCircuitSeqId;
   } m_gate;
};

/** Assignment operation. */
class FALCON_DYN_CLASS ExprAssign: public BinaryExpression
{
public:
   inline ExprAssign( Expression* op1, Expression* op2 ):
      BinaryExpression( t_assign, op1, op2 )
   {
      op1->setLValue();
   }

   inline ExprAssign( const ExprAssign& other ):
      BinaryExpression( other )
   {
   }

   inline virtual ExprAssign* clone() const { return new ExprAssign( *this ); }

   virtual bool simplify( Item& value ) const;
   virtual void toString( String& ) const;

   inline virtual bool isStandAlone() const { return true; }
   virtual void precompile( PCode* pcode ) const;

protected:
   inline ExprAssign():
      BinaryExpression( t_assign ) {}

   friend class ExprFactory;
};

/** Unary negative. */
class FALCON_DYN_CLASS ExprNeg: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprNeg, t_neg );
};



/** Math sum. */
class FALCON_DYN_CLASS ExprPlus: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprPlus, t_plus );
};

/** Less than operator. */
class FALCON_DYN_CLASS ExprLT: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprLT, t_lt );
};


/** Function call. */
class FALCON_DYN_CLASS ExprCall: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprCall, t_funcall );
   virtual ~ExprCall();

   ExprCall& addParameter( Expression* );

   int paramCount() const { return m_params.size(); }
   Expression* getParam( int n ) const;

   inline virtual bool isStandAlone() const { return true; }

   void precompile( PCode* pcode ) const;

private:
   std::vector<Expression*> m_params;
};


#if 0





/** Math subtraction. */
class FALCON_DYN_CLASS ExprMinus: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprMinus, t_minus );
};


/** Math multiply. */
class FALCON_DYN_CLASS ExprTimes: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprTimes, t_times );
};

/** Math divide. */
class FALCON_DYN_CLASS ExprDiv: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprDiv, t_divide );
};

/** Math modulo. */
class FALCON_DYN_CLASS ExprMod: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprMod, t_modulo );
};

/** Math power. */
class FALCON_DYN_CLASS ExprPow: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprPow, t_power );
};


/** Greater than operator. */
class FALCON_DYN_CLASS ExprGT: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprGT, t_gt );
};

/** Greater than or equal to operator. */
class FALCON_DYN_CLASS ExprGE: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprGE, t_ge );
};


/** Less than or equal to operator. */
class FALCON_DYN_CLASS ExprLE: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprLE, t_le );
};

/** Equal to operator. */
class FALCON_DYN_CLASS ExprEQ: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprEQ, t_eq );
};

/** Exactly equal to operator. */
class FALCON_DYN_CLASS ExprEEQ: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprEEQ, t_exeq );
};

/** Not equal to operator. */
class FALCON_DYN_CLASS ExprNE: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprNE, t_neq );
};



To be dealt with later

/** Math unary increment prefix. */
class FALCON_DYN_CLASS ExprPreInc: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPreInc, t_pre_inc );
   inline virtual bool isStandAlone() const { return true; }
};

/** Math unary increment prefix. */
class FALCON_DYN_CLASS ExprPostInc: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPostInc, t_post_inc );
   inline virtual bool isStandAlone() const { return true; }
};

/** Math unary decrement prefix. */
class FALCON_DYN_CLASS ExprPreDec: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPreDec, t_pre_dec );
   inline virtual bool isStandAlone() const { return true; }
};

/** Math unary decrement postfix. */
class FALCON_DYN_CLASS ExprPostDec: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprPostDec, t_post_dec );
   inline virtual bool isStandAlone() const { return true; }
};

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

/** Fast if -- ternary conditional operator. */
class FALCON_DYN_CLASS ExprIIF: public TernaryExpression
{
public:
   FALCON_TERNARY_EXPRESSION_CLASS_DECLARATOR( ExprIIF, t_iif );

   /** Check if the and expression can stand alone.
    *
    * An "?" expression can stand alone if the second AND third operand are standalone.
    */
   inline virtual bool isStandAlone() const {
      return m_second->isStandAlone() && m_third->isStandAlone();
   }
};

/** Dot accessor. */
class FALCON_DYN_CLASS ExprDot: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprIIF, t_obj_access );
};



/** Index accessor. */
class FALCON_DYN_CLASS ExprIndex: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprIndex, t_array_access );
};

/** Special string Index accessor. */
class FALCON_DYN_CLASS ExprStarIndex: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprIndex, t_array_byte_access );
};

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


/** Auto-add operation. */
class FALCON_DYN_CLASS ExprAutoAdd: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprAutoAdd, t_aadd );
   inline virtual bool isStandAlone() const { return true; }
};

/** Auto-subtract operation. */
class FALCON_DYN_CLASS ExprAutoSub: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprAutoSub, t_asub );
   inline virtual bool isStandAlone() const { return true; }
};

/** Auto-multiply operation. */
class FALCON_DYN_CLASS ExprAutoMul: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprAutoMul, t_amul );
   inline virtual bool isStandAlone() const { return true; }
};


/** Auto-divide operation. */
class FALCON_DYN_CLASS ExprAutoDiv: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprAutoDiv, t_adiv );
   inline virtual bool isStandAlone() const { return true; }
};

/** Auto-module operation. */
class FALCON_DYN_CLASS ExprAutoMod: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprAutoMod, t_amod );
   inline virtual bool isStandAlone() const { return true; }
};

/** Auto-power operation. */
class FALCON_DYN_CLASS ExprAutoPow: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( ExprAutoPow, t_apow );
   inline virtual bool isStandAlone() const { return true; }
};


/** Fast-evaluate expression. */
class FALCON_DYN_CLASS ExprEval: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( ExprEval, t_eval );
   inline virtual bool isStandAlone() const { return true; }
};


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

#endif
}

#endif

/* end of expression.h */

