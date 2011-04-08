/*
   FALCON - The Falcon Programming Language.
   FILE: parser/rule.cpp

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 21:17:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/rule.h>
#include <falcon/string.h>

namespace Falcon {
namespace Parser {

Rule::Maker& Rule::Maker::t( const Token& t )
{
}

Rule::Rule( const String& name, Apply app )
{
}
 
Rule::Rule( const Maker& m )
{
}


Rule( const Rule& m )
{
}

virtual ~Rule()
{
}

Rule& t( const Token& t )
{
}


Rule::t_matchType match( const Parser& p )
{
}


}
}

/* end of parser/rule.cpp */
