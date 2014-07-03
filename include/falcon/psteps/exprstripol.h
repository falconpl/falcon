/*
   FALCON - The Falcon Programming Language.
   FILE: exprstripol.h

   Syntactic tree item definitions -- String interpolation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 31 Jan 2013 19:30:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRSTRIPOL_H
#define FALCON_EXPRSTRIPOL_H

#include <falcon/expression.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>
#include <falcon/mt.h>

namespace Falcon {

class StrIPolData;

/** String interpolation.

 */
class FALCON_DYN_CLASS ExprStrIPol: public UnaryExpression
{
public:
   FALCON_UNARY_EXPRESSION_CLASS_DECLARATOR_EX( ExprStrIPol, expr_stripol, \
            m_data = 0; \
            m_bTestExpr = true;\
            );

private:
   mutable StrIPolData* m_data;

   void handleStaticInterpolated( const String &str, VMContext *ctx ) const;
   void handleDynamicInterpolated( const String &str, VMContext *ctx ) const;

   class FALCON_DYN_CLASS PStepIPolData: public PStep
   {
   public:
      PStepIPolData(){apply = apply_;}
      virtual ~PStepIPolData() {}
      static void apply_( const PStep*, VMContext* ctx );
      virtual void describeTo( String& desc ) const
      {
         desc = "ExprStrIPol::PstepIPolData";
      }
   };

   mutable PStepIPolData m_pstepIPolData;
   mutable Mutex m_mtx;
   mutable bool m_bTestExpr;
};

}

#endif	/* FALCON_EXPRSTRIPOL_H */

/* end of exprneg.h */
