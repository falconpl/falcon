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

void apply_Atom_Int ( const NonTerminal&, Parser& p );
void apply_Atom_Float ( const NonTerminal&, Parser& p );
void apply_Atom_Name ( const NonTerminal&, Parser& p );
void apply_Atom_Pure_Name ( const NonTerminal&, Parser& p );
void apply_Atom_String ( const NonTerminal&, Parser& p );
void apply_Atom_RString ( const NonTerminal&, Parser& p );
void apply_Atom_IString ( const NonTerminal&, Parser& p );
void apply_Atom_MString ( const NonTerminal&, Parser& p );
void apply_Atom_False ( const NonTerminal&, Parser& p );
void apply_Atom_True ( const NonTerminal&, Parser& p );
void apply_Atom_Self ( const NonTerminal&, Parser& p );
void apply_Atom_FSelf ( const NonTerminal&, Parser& p );
void apply_Atom_Init ( const NonTerminal&, Parser& p );
void apply_Atom_Nil ( const NonTerminal&, Parser& p );

void apply_expr_atom( const NonTerminal&, Parser& p );
}

#endif

/* end of parser_atom.h */
