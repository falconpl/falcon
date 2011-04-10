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

Parsing::Terminal t_dot("DOT");

Parsing::Terminal t_openpar("(");
Parsing::Terminal t_closepar(")");
Parsing::Terminal t_opensquare("[");
Parsing::Terminal t_closesquare("]");
Parsing::Terminal t_opengraph("{");
Parsing::Terminal t_closegraph("}");

Parsing::Terminal t_plus("+");
Parsing::Terminal t_minus("-");
Parsing::Terminal t_times("*");
Parsing::Terminal t_divide("/");
Parsing::Terminal t_modulo("%");
Parsing::Terminal t_pow("**");

Parsing::Terminal t_token_as("as");
Parsing::Terminal t_token_eq("eq");
Parsing::Terminal t_token_if("if");
Parsing::Terminal t_token_in("in");
Parsing::Terminal t_token_or("or");
Parsing::Terminal t_token_to("to");

Parsing::Terminal t_token_not("not");
Parsing::Terminal t_token_end("end");
Parsing::Terminal t_token_nil("nil");

}

/* end of sourcetokens.cpp */
