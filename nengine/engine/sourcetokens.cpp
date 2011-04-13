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

Parsing::Terminal& t_dot()
{
  static Parsing::Terminal value("DOT");
  return value;
}

Parsing::Terminal& t_openpar()
{
  static Parsing::Terminal value("(");
  return value;
}
Parsing::Terminal& t_closepar()
{
  static Parsing::Terminal value(")");
  return value;
}
Parsing::Terminal& t_opensquare()
{
  static Parsing::Terminal value("[");
  return value;
}
Parsing::Terminal& t_closesquare()
{
  static Parsing::Terminal value("]");
  return value;
}
Parsing::Terminal& t_opengraph()
{
  static Parsing::Terminal value("{");
  return value;
}
Parsing::Terminal& t_closegraph()
{
  static Parsing::Terminal value("}");
  return value;
}

Parsing::Terminal& t_plus()
{
  static Parsing::Terminal value("+");
  return value;
}
Parsing::Terminal& t_minus()
{
  static Parsing::Terminal value("-");
  return value;
}
Parsing::Terminal& t_times()
{
  static Parsing::Terminal value("*");
  return value;
}
Parsing::Terminal& t_divide()
{
  static Parsing::Terminal value("/");
  return value;
}
Parsing::Terminal& t_modulo()
{
  static Parsing::Terminal value("%");
  return value;
}
Parsing::Terminal& t_pow()
{
  static Parsing::Terminal value("**");
  return value;
}
Parsing::Terminal& t_token_as()
{
  static Parsing::Terminal value("as");
  return value;
}
Parsing::Terminal& t_token_eq()
{
  static Parsing::Terminal value("eq");
  return value;
}
Parsing::Terminal& t_token_if()
{
  static Parsing::Terminal value("if");
  return value;
}
Parsing::Terminal& t_token_in()
{
  static Parsing::Terminal value("in");
  return value;
}
Parsing::Terminal& t_token_or()
{
  static Parsing::Terminal value("or");
  return value;
}
Parsing::Terminal& t_token_to()
{
  static Parsing::Terminal value("to");
  return value;
}
Parsing::Terminal& t_token_not()
{
  static Parsing::Terminal value("not");
  return value;
}
Parsing::Terminal& t_token_end()
{
  static Parsing::Terminal value("end");
  return value;
}
Parsing::Terminal& t_token_nil()
{
  static Parsing::Terminal value("nil");
  return value;
}

}

/* end of sourcetokens.cpp */
