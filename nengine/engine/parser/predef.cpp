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
namespace Parser {

TInt t_int;
TFloat t_float;
TString t_string;
TName t_name;
TEol t_eol;
TEof t_eof;

}
}

/* end of parser/predef.cpp */
