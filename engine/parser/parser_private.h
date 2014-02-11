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
#define _FALCON_PARSER_PRIVATE_H_

#include <falcon/parser/parser.h>
#include <deque>
#include <vector>
#include <map>
#include <list>

namespace Falcon {
namespace Parsing {

class Rule;


class Parser::Private
{
   friend class Parser;
   friend class Rule;

   // list of errors found while parsing.
   typedef std::deque<Parser::ErrorDef> ErrorList;
   ErrorList m_lErrors;

   // stack of lexers currently providing tokens.
   typedef std::deque<Lexer*> LexerStack;
   LexerStack m_lLexers;

   // position of the next takeable token for getNextToken
   size_t m_nextTokenPos;

   // Map of existing parsing states.
   typedef std::map<String, NonTerminal*> StateMap;
   StateMap m_states;

   typedef std::vector<const Rule*> RulePath;
   
   class ParseFrame {
   public:
      const NonTerminal* m_owningToken;

      // depth of the stack at this frame.
      int m_nStackDepth;

      // Rule being tested in m_owningToken->term(...);
      int m_hypotesis;
      // Current token in rule being tested.
      int m_hypToken;

      // Lowest priority of the frame.
      int m_prio;

      // position in the frame (relative to stakcDepth) if the token NEXT to the lowest prio.
      int m_prioPos;

      // Token that limits the error recovery mode (if any)
      const Token* m_limitToken;

      // True if the frame has right-associativity
      bool m_bRA;

      // True when the frame is in error recovery mode.
      bool m_bErrorMode;
      

      ParseFrame( const NonTerminal* nt=0, int nd=0 ):
         m_owningToken(nt),
         m_nStackDepth(nd),
         m_hypotesis(0),
         m_hypToken(0),
         m_prio(0),
         m_prioPos(0),
         m_limitToken(0),
         m_bRA(false),
         m_bErrorMode(false)
      {}
         
      virtual ~ParseFrame();
   };

    // stack of read tokens.
   typedef std::vector<TokenInstance*> TokenStack;
   TokenStack* m_tokenStack;

   typedef std::deque<ParseFrame> FrameStack;
   FrameStack* m_pframes;
   FrameStack* m_pErrorFrames;

   class StateFrame {
   public:
      NonTerminal* m_state;
      TokenStack m_tokenStack;

      FrameStack m_pframes;
      FrameStack m_pErrorFrames;

      Parser::StateFrameFunc m_cbfunc;
      void* m_cbdata;
      int m_id;
      int m_appliedRules;

      StateFrame( NonTerminal* s );
      ~StateFrame();
   };
   
   // Currently active parsing states.
   typedef std::list<StateFrame*> StateStack;
   StateStack m_lStates;
   int m_stateFrameID;
   int m_lastLine;


   Private();
   ~Private();
   
   void clearTokens();
   void clearStates();
};

}
}

#endif	/* _FALCON_PARSER_PRIVATE_H_ */

/* end of parser_private.h */
