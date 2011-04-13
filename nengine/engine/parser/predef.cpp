/*
   FALCON - The Falcon Programming Language.
   FILE: parser/predef.cpp

   Instantation of predefined tokens.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 20:15:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/tint.h>
#include <falcon/parser/tfloat.h>
#include <falcon/parser/tstring.h>
#include <falcon/parser/tname.h>
#include <falcon/parser/teol.h>
#include <falcon/parser/teof.h>

namespace Falcon {
namespace Parsing {

TInt& t_int()
{
  static TInt value;
  return value;
}
TFloat& t_float()
{
  static TFloat value;
  return value;
}
TString& t_string()
{
  static TString value;
  return value;
}
TName& t_name()
{
  static TName value;
  return value;
}
TEol& t_eol()
{
  static TEol value;
  return value;
}
TEof& t_eof()
{
  static TEof value;
  return value;
}

}
}

/* end of parser/predef.cpp */
