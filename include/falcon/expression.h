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
#include <falcon/treestep.h>
#include <falcon/sourceref.h>

namespace Falcon
{

class DataReader;
class DataWriter;
class PCode;
class PseudoFunction;
class Symbol;

/** Pure abstract class representing a Falcon expression.
   Base for all the expressions in the language.
 @see TreeStep
 */
class FALCON_DYN_CLASS Expression: public TreeStep
{
public:   
   
   Expression( const Expression &other );
   virtual ~Expression();

   typedef enum {
      e_trait_none,
      e_trait_symbol,
      e_trait_value,
      e_trait_assignable,
      e_trait_inheritance,
      e_trait_composite,
      e_trait_unquote
   }
   t_trait;
   
   // TODO: Rename in trait()
   t_trait trait() const { return m_trait; }
   

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


   /** Evaluates the expression when all its components are static.
    * @Return true if the expression can be simplified, false if it's not statitc
    *
    * Used during compilation to simplify static expressions, that is,
    * reducing expressions at compile time.
    */
   virtual bool simplify( Item& result ) const = 0;   
      
   virtual Expression* clone() const = 0;

protected:
   
   Expression( int line = 0, int chr = 0  ):
      TreeStep( e_cat_expression, line, chr ),
      m_pstep_lvalue(0),
      m_trait( e_trait_none )
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
   t_trait m_trait;
};


/** Pure abstract Base unary expression. */
class FALCON_DYN_CLASS UnaryExpression: public Expression
{
public:
   inline UnaryExpression( Expression* op1, int line = 0, int chr = 0 ):
      Expression( line, chr ),
      m_first( op1 )
   {
      m_first->setParent(this);
   }
      
   inline UnaryExpression( int line = 0, int chr = 0 ):
      Expression( line, chr ),
      m_first( 0 )
   {}

   UnaryExpression( const UnaryExpression& other );
   virtual ~UnaryExpression();

   virtual bool isStatic() const;

   Expression *first() const { return m_first; }
   void first( Expression *f ) { 
      if ( f->setParent(this) )
      {
         delete m_first; 
         m_first= f; 
      }
   }

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   
   
protected:
   Expression* m_first;
};


/** Pure abstract Base binary expression. */
class FALCON_DYN_CLASS BinaryExpression: public Expression
{
public:

   inline BinaryExpression( int line=0, int chr =0 ):
      Expression( line, chr ),
      m_first( 0 ),
      m_second( 0 )
   {}
      
   inline BinaryExpression( Expression* op1, Expression* op2, int line = 0, int chr = 0 ):
      Expression( line, chr ),
      m_first( op1 ),
      m_second( op2 )
   {
      m_first->setParent(this);
      m_second->setParent(this);
   }

   BinaryExpression( const BinaryExpression& other );
   virtual ~BinaryExpression();

   Expression *first() const { return m_first; }
   void first( Expression *f ) { 
      if ( f->setParent(this) )
      {
         delete m_first; 
         m_first= f; 
      }
   }
   Expression *second() const { return m_second; }
   void second( Expression *f ) { 
      if ( f->setParent(this) )
      {
         delete m_second; 
         m_second= f; 
      }
   }

   virtual bool isStatic() const;

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   
protected:
   
   Expression* m_first;
   Expression* m_second;
   
};


/** Pure abstract Base ternary expression. */
class FALCON_DYN_CLASS TernaryExpression: public Expression
{
public:
  
   inline TernaryExpression( Expression* op1, Expression* op2, Expression* op3, int line = 0, int chr = 0 ):
      Expression( line, chr ),
      m_first( op1 ),
      m_second( op2 ),
      m_third( op3 )
   {
      m_first->setParent(this);
      m_second->setParent(this);
      m_third->setParent(this);
   }

   inline TernaryExpression( int line = 0, int chr = 0 ):
      Expression( line, chr ),
      m_first( 0 ),
      m_second( 0 ),
      m_third( 0 )
   {}
      
   TernaryExpression( const TernaryExpression& other );

   virtual ~TernaryExpression();
   virtual bool isStatic() const;

   Expression *first() const { return m_first; }
   void first( Expression *f ) { 
      if ( f->setParent(this) )
      {
         delete m_first; 
         m_first= f; 
      }
   }
   Expression *second() const { return m_second; }
   void second( Expression *f ) { 
      if ( f->setParent(this) )
      {
         delete m_second; 
         m_second= f; 
      }
   }
   Expression *third() const { return m_third; }
   void third( Expression *f ) { 
      if ( f->setParent(this) )
      {
         delete m_third; 
         m_third= f; 
      }
   }
   
   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   
protected:
   Expression* m_first;
   Expression* m_second;   
   Expression* m_third;
};

//==============================================================
// Many operators take this form:
//

#define FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( class_name, handler ) \
   inline class_name( Expression* op1, int line=0, int chr=0 ): \
            UnaryExpression( op1, line, chr ) { FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_;} \
   inline class_name(int line=0, int chr=0): \
            UnaryExpression(line,chr) { FALCON_DECLARE_SYN_CLASS( handler );  apply = apply_; }\
   inline class_name( const class_name& other ):\
            UnaryExpression( other ) {FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_;} \
   inline ~class_name() {}\
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual void describeTo( String&, int depth = 0 ) const;

#define FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( class_name, handler ) \
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, handler, )

#define FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, handler, extended_constructor ) \
   inline class_name( Expression* op1, Expression* op2, int line=0, int chr=0 ): \
            BinaryExpression( op1, op2, line, chr ) { FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; extended_constructor} \
   inline class_name(int line=0, int chr=0): \
            BinaryExpression(line,chr) { FALCON_DECLARE_SYN_CLASS( handler );  apply = apply_; extended_constructor}\
   inline class_name( const class_name& other ):\
            BinaryExpression( other ) {FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; extended_constructor} \
   inline ~class_name() {}\
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual void describeTo( String&, int depth=0 ) const;\
   public:

#define FALCON_TERNARY_EXPRESSION_CLASS_DECLARATOR( class_name, handler ) \
   inline class_name( Expression* op1, Expression* op2, Expression* op3, int line=0, int chr=0 ): \
            TernaryExpression( op1, op2, op3, line, chr ) { FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_;} \
   inline class_name(int line=0, int chr=0): \
            TernaryExpression(line,chr) { FALCON_DECLARE_SYN_CLASS( handler );  apply = apply_; }\
   inline class_name( const class_name& other ):\
            TernaryExpression( other ) {FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; } \
   inline ~class_name() {}\
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual void describeTo( String&, int depth = 0 ) const;\
   public:

//==============================================================


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


#endif

}

#endif

/* end of expression.h */
