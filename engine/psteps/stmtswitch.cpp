/*
   FALCON - The Falcon Programming Language.
   FILE: stmtswitch.cpp

   Syntactic tree item definitions -- switch statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 May 2012 16:33:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/psteps/stmtswitch.cpp"

#include <falcon/expression.h>
#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/syntree.h>
#include <falcon/psteps/stmtswitch.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/itemarray.h>

#include <map>
#include <set>
#include <vector>
#include <algorithm>

namespace Falcon {

StmtSwitch::StmtSwitch( int32 line, int32 chr ):
   SwitchlikeStatement( line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_switch );
   apply = apply_;
}


StmtSwitch::StmtSwitch( Expression* expr, int32 line, int32 chr ):
   SwitchlikeStatement( expr, line, chr )
{
   FALCON_DECLARE_SYN_CLASS( stmt_switch );
   apply = apply_;
}

StmtSwitch::StmtSwitch( const StmtSwitch& other ):
   SwitchlikeStatement( other )
{
   apply = apply_;
}


StmtSwitch::~StmtSwitch()
{
}


void StmtSwitch::describeTo( String& tgt, int depth ) const
{
   if( m_expr == 0 )
   {
      tgt = "<Blank switch>";
      return;
   }
   
   String prefix = String(" ").replicate( depth * depthIndent );
   tgt = prefix + "switch " + m_expr->describe() +"\n";
}



void StmtSwitch::apply_( const PStep* ps, VMContext* ctx )
{
   const StmtSwitch* self = static_cast<const StmtSwitch*>(ps);

   CodeFrame& cf = ctx->currentCode();
   // first time around? -- call the expression.
   switch( cf.m_seqId )
   {
      case 0:
         cf.m_seqId = 1;
         if( ctx->stepInYield( self->m_expr, cf ) )
         {
            return;
         }
         /* no break */
        
      case 1:         
         {
            SynTree* selectedBlock = self->findBlockForItem( ctx->topData(), ctx );
            // found?
            if( selectedBlock != 0 ) {
               // -- we're done.
               ctx->popCode();
               ctx->popData();
               ctx->stepIn( selectedBlock );
               return;
            }
         }
         break;
   }

   // nope, we didn't find it.
   // anyway, we're done...
   ctx->popCode();
   // leave the expression value on the stack.
}

}

/* end of stmtswitch.cpp */
