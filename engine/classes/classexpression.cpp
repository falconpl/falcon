/*
   FALCON - The Falcon Programming Language.
   FILE: classexpression.cpp

   Base class for expression PStep handlers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/classes/classexpression.h>
#include <falcon/classes/classtreestep.h>

#include <falcon/vmcontext.h>
#include <falcon/expression.h>

#include "falcon/stdsteps.h"

namespace Falcon {

ClassExpression::ClassExpression( ClassTreeStep* parent ):
   DerivedFrom( parent, "Expression" )
{}
   
ClassExpression::~ClassExpression(){}

void ClassExpression::op_call( VMContext* ctx, int pcount, void* instance ) const
{
   static StdSteps* steps = Engine::instance()->stdSteps();

   // Remove the top of the stack because our expression will do its own.
   Expression* self = static_cast<Expression*>(instance);
   VarMap* st = self->varmap();
   // Do we have a symbol table?
   if( st == 0 )
   {
      ctx->popData(pcount+1);
      //ctx->addLocalFrame(0,0);
      ctx->pushCode(&steps->m_localFrame);
   }
   else {
      ctx->addLocalFrame(st,pcount);
   }
   
   ctx->pushCode( self );
}

}

/* end of classexpression.cpp */
