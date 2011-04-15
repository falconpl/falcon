/*
   FALCON - The Falcon Programming Language.
   FILE: parser/nonterminal.h

   Token representing a non-terminal grammar symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 12:54:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_NONTERMINAL_H_
#define	_FALCON_PARSER_NONTERMINAL_H_

#include <falcon/setup.h>
#include <falcon/parser/token.h>
#include <falcon/parser/matchtype.h>

namespace Falcon {
namespace Parsing {

class Rule;
class Parser;

/** NonTerminal parser symbols.

 NonTerminal symbols are symbol built up from simpler Terminal or NonTerminal
 symbols through one or more rules.

 To create a NonTerminal:
    @code
    NonTerminal SomeEntity;
    SomeEntity \<\< "name" \<\< terminal1 << terminal2 ... << terminalN;

    // or
    NonTerminal SomeEntity( "Name" );
    SomeEntity \<\< terminal1 << terminal2 ... << terminalN;
    @endcode

 */
class FALCON_DYN_CLASS NonTerminal: public Token
{
public:
   NonTerminal();

   /** Normal constructor. */
   NonTerminal(const String& name);

   virtual ~NonTerminal();

   /** Adds a rule to this non-terminal symbol.
    \return this symbol.
    */
   NonTerminal& r(Rule& rule);

   /** Return true if a match is confirmed.
    \param p The parser on which the matches are checked.
    \return true if any of the rules in the  NonTerminal is matched. */
   t_matchType match( Parser& parser );

   inline NonTerminal& operator <<( Rule& rule ) { return r(rule); }
   inline NonTerminal& operator <<( const String& n ) {
      name(n);
      return *this; 
   }

private:
   class Private;
   Private* _p;
};

}
}

#endif	/* _FALCON_PARSER_NONTERMINAL_H_ */

/* end of parser/nonterminal.h */
