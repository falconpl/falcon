/*
   FALCON - The Falcon Programming Language.
   FILE: sourcetokens.cpp

   Definition of grammar tokens known by the Falcon source parser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Apr 2011 23:13:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/sourcetokens.h>

namespace Falcon {

Parser::Terminal t_dot("DOT");

Parser::Terminal t_openpar("(");
Parser::Terminal t_closepar(")");
Parser::Terminal t_opensquare("[");
Parser::Terminal t_closesquare("]");
Parser::Terminal t_opengraph("{");
Parser::Terminal t_closegraph("}");

Parser::Terminal t_plus("+");
Parser::Terminal t_minus("-");
Parser::Terminal t_times("*");
Parser::Terminal t_divide("/");
Parser::Terminal t_modulo("%");
Parser::Terminal t_pow("**");

Parser::Terminal t_token_as("as");
Parser::Terminal t_token_eq("eq");
Parser::Terminal t_token_if("if");
Parser::Terminal t_token_in("in");
Parser::Terminal t_token_or("or");
Parser::Terminal t_token_to("to");

Parser::Terminal t_token_not("not");
Parser::Terminal t_token_end("end");
Parser::Terminal t_token_nil("nil");

}

/* end of sourcetokens.cpp */
