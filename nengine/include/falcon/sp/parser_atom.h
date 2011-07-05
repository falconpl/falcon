/*
   FALCON - The Falcon Programming Language.
   FILE: parser_atom.h

   Parser for Falcon source files -- handler for atoms and values
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 18:24:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PARSER_ATOM_H_
#define _FALCON_SP_PARSER_ATOM_H_

#include <falcon/setup.h>

namespace Falcon {

namespace Parsing
{
class Rule;
class Parser;
}

using namespace Parsing;

void apply_Atom_Int ( const Rule&, Parser& p );
void apply_Atom_Float ( const Rule&, Parser& p );
void apply_Atom_Name ( const Rule&, Parser& p );
void apply_Atom_String ( const Rule&, Parser& p );
void apply_Atom_False ( const Rule&, Parser& p );
void apply_Atom_True ( const Rule&, Parser& p );
void apply_Atom_Self ( const Rule&, Parser& p );
void apply_Atom_Nil ( const Rule&, Parser& p );

void apply_expr_atom( const Rule&, Parser& p );
}

#endif

/* end of parser_atom.h */
