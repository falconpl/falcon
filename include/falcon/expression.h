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
      e_trait_unquote,
      e_trait_vectorial,
      e_trait_case,
      e_trait_tree,
      e_trait_named
   }
   t_trait;
   
   // TODO: Rename in trait()
   t_trait trait() const { return m_trait; }
      
   /** True if an automatic definition of this expression can define symbols. */
   virtual bool fullDefining() { return false; }

   virtual Expression* clone() const = 0;

protected:
   
   Expression( int line = 0, int chr = 0  ):
      TreeStep( e_cat_expression, line, chr ),
      m_trait( e_trait_none )
   {}
      
   t_trait m_trait;
};


/** Pure abstract Base unary expression. */
class FALCON_DYN_CLASS UnaryExpression: public Expression
{
public:
   inline UnaryExpression( TreeStep* op1, int line = 0, int chr = 0 ):
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

   TreeStep *first() const { return m_first; }
   void first( TreeStep *f ) {
      if ( f->setParent(this) )
      {
         dispose( m_first );
         m_first= f; 
      }
   }

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   
   virtual const String& exprName() const = 0;
   void render( TextWriter* tw, int depth ) const;
   
protected:
   TreeStep* m_first;
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
      
   inline BinaryExpression( TreeStep* op1, TreeStep* op2, int line = 0, int chr = 0 ):
      Expression( line, chr ),
      m_first( op1 ),
      m_second( op2 )
   {
      m_first->setParent(this);
      m_second->setParent(this);
   }

   BinaryExpression( const BinaryExpression& other );
   virtual ~BinaryExpression();

   TreeStep *first() const { return m_first; }
   void first( TreeStep *f ) {
      if ( f->setParent(this) )
      {
         dispose( m_first );
         m_first= f; 
      }
   }
   TreeStep *second() const { return m_second; }
   void second( TreeStep *f ) {
      if ( f->setParent(this) )
      {
         dispose( m_second );
         m_second= f; 
      }
   }

   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   
   virtual const String& exprName() const = 0;
   void render( TextWriter* tw, int depth ) const;
protected:
   
   TreeStep* m_first;
   TreeStep* m_second;
   
};


/** Pure abstract Base ternary expression. */
class FALCON_DYN_CLASS TernaryExpression: public Expression
{
public:
  
   inline TernaryExpression( TreeStep* op1, TreeStep* op2, TreeStep* op3, int line = 0, int chr = 0 ):
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

   TreeStep *first() const { return m_first; }
   void first( TreeStep *f ) {
      if ( f->setParent(this) )
      {
         dispose( m_first );
         m_first= f; 
      }
   }
   TreeStep *second() const { return m_second; }
   void second( TreeStep *f ) {
      if ( f->setParent(this) )
      {
         dispose( m_second );
         m_second= f; 
      }
   }
   TreeStep *third() const { return m_third; }
   void third( TreeStep *f ) {
      if ( f->setParent(this) )
      {
         dispose( m_third );
         m_third= f; 
      }
   }
   
   virtual int32 arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   
protected:
   TreeStep* m_first;
   TreeStep* m_second;
   TreeStep* m_third;
};

//==============================================================
// Many operators take this form:
//

#define FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR( class_name, handler ) \
         FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, handler, )

#define FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, handler, extend_constructor ) \
   inline class_name( TreeStep* op1, int line=0, int chr=0 ): \
            UnaryExpression( op1, line, chr ) { FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; extend_constructor} \
   inline class_name(int line=0, int chr=0): \
            UnaryExpression(line,chr) { FALCON_DECLARE_SYN_CLASS( handler );  apply = apply_; extend_constructor }\
   inline class_name( const class_name& other ):\
            UnaryExpression( other ) {FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; extend_constructor} \
   inline ~class_name() {}\
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual const String& exprName() const;


#define FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR( class_name, handler ) \
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, handler, )

#define FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR_EX( class_name, handler, extended_constructor ) \
   inline class_name( TreeStep* op1, TreeStep* op2, int line=0, int chr=0 ): \
            BinaryExpression( op1, op2, line, chr ) { FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; extended_constructor} \
   inline class_name(int line=0, int chr=0): \
            BinaryExpression(line,chr) { FALCON_DECLARE_SYN_CLASS( handler );  apply = apply_; extended_constructor}\
   inline class_name( const class_name& other ):\
            BinaryExpression( other ) {FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; extended_constructor} \
   inline ~class_name() {}\
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual const String& exprName() const; \
   public:

#define FALCON_TERNARY_EXPRESSION_CLASS_DECLARATOR( class_name, handler ) \
   inline class_name( TreeStep* op1, TreeStep* op2, TreeStep* op3, int line=0, int chr=0 ): \
            TernaryExpression( op1, op2, op3, line, chr ) { FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_;} \
   inline class_name(int line=0, int chr=0): \
            TernaryExpression(line,chr) { FALCON_DECLARE_SYN_CLASS( handler );  apply = apply_; }\
   inline class_name( const class_name& other ):\
            TernaryExpression( other ) {FALCON_DECLARE_SYN_CLASS( handler ); apply = apply_; } \
   inline ~class_name() {}\
   inline virtual class_name* clone() const { return new class_name( *this ); } \
   virtual bool simplify( Item& value ) const; \
   static void apply_( const PStep*, VMContext* ctx ); \
   virtual void render( TextWriter* tw, int depth ) const;\
   public:


}

#endif

/* end of expression.h */
