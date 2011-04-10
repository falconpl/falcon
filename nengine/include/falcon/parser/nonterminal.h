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
namespace Parser {

class Rule;
class Parser;

/** NonTerminal parser symbols.

 NonTerminal symbols are symbol built up from simpler Terminal or NonTerminal
 symbols through one or more rules.

 */
class FALCON_DYN_CLASS NonTerminal: public Token
{
private:
    class Private;

public:
   /** Support for variable parameter constructor idiom.
    To create a NonTerminal:
    Rule r1 = Rule::Maker("name", R_Apply ).t( terminal1 ).t( NonTerminal2 ).t( t_EOL ) )....;

    NonTerminal t = NonTerminal::Maker( "Subname" ).r( r1 ).r(...)...;

    */
   class Maker
   {
      friend class NonTerminal;
      Maker( const String& name );
      ~Maker();

      /** Adds a rule to this nonterminal.
       \return an instance of this item.
       */
      Maker& r( Rule& t );

   private:
      const String& m_name;

      // inner tokens.
      mutable NonTerminal::Private* _p;
   };

   /** Normal constructor. */
   NonTerminal(const String& name);

   /** Constructor using the Maker assignment. */
   NonTerminal(const Maker& name);

   virtual ~NonTerminal();

   /** Adds a rule to this non-terminal symbol.
    \return this symbol.
    */
   NonTerminal& r(Rule& rule);

   /** Return true if a match is confirmed.
    \param p The parser on which the matches are checked.
    \return true if any of the rules in the  NonTerminal is matched. */
   t_matchType match( Parser& parser );

private:
   Private* _p;
};

}
}

#endif	/* _FALCON_PARSER_NONTERMINAL_H_ */

/* end of parser/nonterminal.h */
