/*
   FALCON - The Falcon Programming Language.
   FILE: exprindex.h

   Syntactic tree item definitions -- Index accessor
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Sat, 30 Jul 2011 16:26:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRINDEX_H_
#define _FALCON_EXPRINDEX_H_

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

namespace Falcon
{

/** Index accessor. */
class FALCON_DYN_CLASS ExprIndex: public BinaryExpression
{
public:
   FALCON_BINARY_EXPRESSION_CLASS_DECLARATOR_EX( ExprIndex, expr_index, \
            m_pstep_lvalue = &m_pslv; \
            m_pslv.m_owner = this; \
            m_trait = e_trait_assignable; );

   virtual void render( TextWriter* tw, int depth ) const;
private:

   /** Step used to SET a value in the array.*/
   class FALCON_DYN_CLASS PstepLValue: public PStep
   {
   public:
      PstepLValue() { apply = apply_; }
      virtual ~PstepLValue() {}
      virtual void describeTo( String& target ) const { target = "Index lvalue"; }
      static void apply_( const PStep*, VMContext* ctx );
   private:
      ExprIndex* m_owner;
      friend class ExprIndex;
   };
   PstepLValue m_pslv;
};

}

#endif

/* end of exprindex.h */
