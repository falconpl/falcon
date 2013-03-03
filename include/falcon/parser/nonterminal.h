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
#define _FALCON_PARSER_NONTERMINAL_H_

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
   NonTerminal(const String& name, bool bRightAssoc = false );

   virtual ~NonTerminal();

   /** Adds a rule to this non-terminal symbol.
    \return this symbol.
    */
   NonTerminal& r(Rule& rule);

   inline NonTerminal& operator <<( Rule& rule ) { return r(rule); }
   inline NonTerminal& operator <<( const String& n ) {
      name(n);
      return *this; 
   }

   /** Callback for parsing error routine.
    \see setErrorHandler
    */
   typedef bool (*ErrorHandler)(const NonTerminal& nt, Parser& p);

   /** Sets the error handler for this routine.
   \param hr An handler routine that is invoked on syntax error.

    Once a route has invariabily pinpointed a non-terminal token as the
    only possible one, in case the none of its rule can be matched as the
    analisys proceeds, an error in the parsing structure is detected.

    Tokens are then back-tracked so that the most specific error-handler that
    has been set the most downwards in the rule hierarcy is found, and that
    is invoked. If the process can't find any error handler up to the state root,
    the gobal parser error handler is invoked.

    A error handler should set an error in the parser through Parser::addError,
    clear the current stack through Parser::simplify and eventually discard
    more incoming tokens up to a simple match point via Parser::resynch. Alternately,
    the current expression may be kept in the stack so that further error detection
    can be performed on the rest of the input.
    
    \see Parser::resynch
    */
   inline NonTerminal& setErrorHandler( ErrorHandler hr ) {
      m_eh = hr;
      return *this;
   }

   inline ErrorHandler errorHandler() const { return m_eh; }
   
   /** Proxy to setErrorHandler. */
   inline NonTerminal& operator << ( ErrorHandler hr ) { return setErrorHandler(hr); }

   bool findPaths( Parser& p ) const;
   void addFirstRule( Parser& p ) const;
   
   int maxArity() const { return m_maxArity; }
   bool isGreedy() const { return m_bGreedy; }
   bool isRecursive() const { return m_bRecursive; }

private:
   ErrorHandler m_eh;
   int m_maxArity;
   bool m_bRecursive;
   bool m_bGreedy;
   
   class Private;
   Private* _p;
};

}
}

#endif	/* _FALCON_PARSER_NONTERMINAL_H_ */

/* end of parser/nonterminal.h */
