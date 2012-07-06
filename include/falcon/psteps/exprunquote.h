/*
   FALCON - The Falcon Programming Language.
   FILE: exprunquote.h

   Syntactic tree item definitions -- Unquote expression (^~)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPR_UNQUOTE_H_
#define FALCON_EXPR_UNQUOTE_H_

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>


namespace Falcon {

/** Unquote expression.
 Unquote expressions (^~symbol) work tightly together with literal expression 
 causing local evaluation to be performed before quoting it.
 
 For instance:
 \code
 expr = {() a+ ^~b + c)
 \endcode
 This causes expr to be filled with an expression "a+X+c", where X is the value
 of "b" expression at the moment it was set in expr. 
 
 This is achieved by creating a "local closure" that stores the value of B
 at the moment of generating the literal expression.
 
 Unquote is a semantic token that can be used as a part of the compiled code
 only. ClassUnquote() class exists to allow the engine to deal with the 
 unquote pstep, but it cannot be invoked nor modified from the outside. However,
 it can be replaced live as any other element of a syntree.
 
 The result of the unquoted expression is actually macro-substituted in the 
 final expression, so
 \code
 function makeExpr()
   b = {()c + d}
   return {() a+ ^~b }
 end
 \endcode
 
 or more interestingly:
 \code
 function makeExpr*( b )
   return {() a+ ^~b }
 end
 
 makeExpr( c + d )
 \endcode
 
 results in expr being "a + c + d".

 Notice that following the closing rules, the unquote expression won't be
 able to close global and exported symbols. Only the local symbol in the 
 symbol tables of the enclosing function hierarcy will be considered.
 
 */
class FALCON_DYN_CLASS ExprUnquote: public Expression
{
public:
   ExprUnquote( int line=0, int chr=0 );
   ExprUnquote( const String& symbol, int line=0, int chr=0 );
   ExprUnquote( const ExprUnquote& other );
   virtual ~ExprUnquote();
   
   inline virtual bool isStandAlone() const { return false; }      
   virtual void describeTo( String& str, int depth ) const;   
   virtual bool simplify(Falcon::Item&) const;
   virtual bool isStatic() const { return false; }
   virtual ExprUnquote* clone() const { return new ExprUnquote(*this); }

   /** Gets the name of the symbol associated with this unquote.
    */
   const String& symbolName() const { return m_symbolName; }
   
   Symbol* symbol() const { return m_dynsym; }

   /** Sets the name of the symbol associated with this unquote.
    */
   void symbolName( const String& s );
   
   /** 
    Gets the Registration ID (pushed parameter in the evaluation).
   */
   int32 regID() const { return m_regID; }
   
   /** 
    Changes the Registration ID (pushed parameter in the evaluation).
   */
   void regID( int32 i ) { m_regID = i; }
   
private:
   static void apply_( const PStep*, VMContext* ctx );
   // registration ID (pushed parameter in the evaluation).
   int32 m_regID;
   String m_symbolName;
   Symbol* m_dynsym;
};

}

#endif

/* end of exprunquote.h */
