/*
   FALCON - The Falcon Programming Language.
   FILE: attribute_helper.cpp

   Little helper for attribute declaration.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Jan 2013 21:00:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_attribute.cpp"

#include <falcon/attribute.h>
#include <falcon/string.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/engine.h>
#include <falcon/mantra.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>

namespace Falcon {

bool attribute_helper(VMContext* vmctx, const String& name, TreeStep* generator, Mantra* target )
{
   static PStep* fillAttribute = &Engine::instance()->stdSteps()->m_fillAttribute;
   static PStep* pnil = &Engine::instance()->stdSteps()->m_pushNil;

   Attribute* attr = target->attributes().add(name);
   if( attr == 0 ) {
      return false;
   }

   if ( generator->category() == TreeStep::e_cat_expression )
   {
      Expression* expr = static_cast<Expression *>(generator);

      if( expr->trait() == Expression::e_trait_value )
      {
         attr->value() = static_cast<ExprValue *>(expr)->item();
         delete expr;
         return true;
      }
      else {

         vmctx->pushData( Item("Attribute", attr) );
         vmctx->pushCode( pnil ); // the parser will want an extra data.
         // we're pretty sure they'll stay alive the whole execution.
         vmctx->pushCode( fillAttribute );
         vmctx->pushCode( generator );
      }
   }

   return true;
}

}

/* end of attribute_helper.cpp */
