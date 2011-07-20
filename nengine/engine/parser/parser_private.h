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
   typedef std::map<String, State*> StateMap;
   StateMap m_states;

   typedef std::vector<const Rule*> RulePath;
   
   class ParseFrame {
   public:
      NonTerminal* m_owningToken;
      // depth of the stack at this frame.
      int m_nStackDepth;
      // highest priority in this stack frame
      int m_nPriority;
      // is this stack frame right associative?        
      bool m_bIsRightAssoc;
      // And this is the position in the stack frame of the highest priority token
      int m_prioFrame;

      RulePath m_path;
      //Alternatives m_candidates;

      ParseFrame( NonTerminal* nt=0, int nd=0 ):
         m_owningToken(nt),
         m_nStackDepth(nd),
         m_nPriority(0),
         m_bIsRightAssoc(false),
         m_prioFrame(0)
      {}

      ~ParseFrame();
   };

    // stack of read tokens.
   typedef std::vector<TokenInstance*> TokenStack;
   TokenStack* m_tokenStack;

   typedef std::list<ParseFrame> FrameStack;
   FrameStack* m_pframes;
   FrameStack* m_pErrorFrames;

   class StateFrame {
   public:
      State* m_state;
      TokenStack m_tokenStack;

      FrameStack m_pframes;
      FrameStack m_pErrorFrames;

      Parser::StateFrameFunc m_cbfunc;
      void* m_cbdata;
      int m_id;
      int m_appliedRules;

      StateFrame( State* s ):
         m_state( s ),
         m_cbfunc( 0 ),
         m_cbdata( 0 ),
         m_id(0),
         m_appliedRules(0)
      {
      }
   };
   
   // Currently active parsing states.
   typedef std::list<StateFrame> StateStack;
   StateStack m_lStates;
   int m_stateFrameID;


   Private();
   ~Private();
   
   void clearTokens();
};

}
}

#endif	/* _FALCON_PARSER_PRIVATE_H_ */

/* end of parser_private.h */
