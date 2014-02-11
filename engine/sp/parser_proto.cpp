/*
   FALCON - The Falcon Programming Language.
   FILE: parser_proto.cpp

   Parser for Falcon source files -- prototype declarations handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:13:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/sp/parser_function.cpp"

#include <falcon/setup.h>
#include <falcon/statement.h>
#include <falcon/error.h>
#include <falcon/parser/parser.h>

#include <falcon/sp/sourceparser.h>
#include <falcon/sp/parsercontext.h>
#include <falcon/sp/parser_deletor.h>
#include <falcon/sp/parser_proto.h>

#include <falcon/psteps/exprproto.h>

#include "private_types.h"

namespace Falcon {

using namespace Parsing;

// temporary statement used to keep track of the forming prototype expression
class StmtTempProto: public Statement
{
public:
   ExprProto* m_forming;

   StmtTempProto():
      Statement( 0,0 )
   {
      // don't record us, we're temp.
      m_discardable = true;
   }

   ~StmtTempProto()
   {
   }
   
   virtual StmtTempProto* clone() const { return 0; }
   virtual void render( TextWriter*, int32 ) const {};
};

/*
static void on_close_proto( void *thing )
{
   // ensure single expressions to be considered returns.
   SourceParser& sp = *static_cast<SourceParser*>(thing);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());

   ctx->
}
*/

void apply_expr_proto( const NonTerminal&, Parser& p)
{
   // << T_OpenProto
   SourceParser& sp = *static_cast<SourceParser*>(&p);
   ParserContext* ctx = static_cast<ParserContext*>(sp.context());

   TokenInstance* ti = TokenInstance::alloc( 0,0, sp.Expr );
   ExprProto* eproto = new ExprProto;
   ti->setValue( eproto, treestep_deletor );
   p.simplify( 1, ti );

   StmtTempProto* proto = new StmtTempProto;
   proto->m_forming = eproto;
   ctx->openBlock( proto, 0 );
   
   //p.pushState( "ProtoDecl", on_close_proto, &p );
   p.pushState( "ProtoDecl" );
}

void apply_proto_prop( const NonTerminal&, Parser& p)
{
    // << T_Name << T_EqSign << Expr << T_EOL
    SourceParser& sp = *static_cast<SourceParser*>(&p);
    ParserContext* ctx = static_cast<ParserContext*>(sp.context());
    StmtTempProto* proto = static_cast<StmtTempProto*>(ctx->currentStmt());
    fassert( proto != 0 );
    fassert( proto->handler() == 0 ); // temporary statements have no class.

    TokenInstance* tname = sp.getNextToken();
    sp.getNextToken();
    TokenInstance* texpr = sp.getNextToken();

    Expression* expr = static_cast<Expression*>(texpr->detachValue());
    proto->m_forming->add( *tname->asString(), expr );

    p.simplify(4);
}

}

/* end of parser_proto.cpp */
