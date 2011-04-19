/*
   FALCON - The Falcon Programming Language.
   FILE: parser/parser_private.h

   Private part of the parser, put in common within the parser engine
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Apr 2011 12:52:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_PRIVATE_H_
#define	_FALCON_PARSER_PRIVATE_H_

#include <falcon/parser/parser.h>
#include <deque>
#include <vector>
#include <map>

namespace Falcon {
namespace Parsing {

class Rule;

class Parser::Private
{
   friend class Parser;
   friend class Rule;

   typedef std::deque<Lexer*> LexerStack;
   typedef std::deque<State*> StateStack;
   typedef std::deque<Parser::ErrorDef*> ErrorList;
   typedef std::map<String, State*> StateMap;

   typedef std::vector<TokenInstance*> TokenStack;
   typedef std::vector<NonTerminal*> Path;
   typedef std::vector<Path*> PathSet;

   StateStack m_lStates;
   TokenStack m_vTokens;
   LexerStack m_lLexers;
   ErrorList m_lErrors;

   TokenInstance* m_nextToken;
   StateMap m_states;

   /** Position used as current starting point in sub-rule matches. */
   size_t m_stackPos;
   size_t m_nextTokenPos;

   Path* m_pCurPath;
   PathSet m_paths;
   PathSet::iterator m_curPathIter;
   
   Private();
   ~Private();
   
   void clearTokens();
   void clearPaths();

   /** Resets the temporary values set in  top-level match. */
   void resetMatch();
};

}
}

#endif	/* _FALCON_PARSER_PRIVATE_H_ */

/* end of parser_private.h */
