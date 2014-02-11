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
#include <falcon/textwriter.h>
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

   /** Callback for parsing error routine.
    \see setErrorHandler
    */
   typedef bool (*ErrorHandler)(const NonTerminal& nt, Parser& p);

   /** Callback for parsing error routine.
       \see setErrorHandler
   */
   typedef void (*Handler)(Parser& p, const NonTerminal& nt);

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
   
   inline NonTerminal& setHandler( Handler hr )
   {
      m_handler = hr;
      return *this;
   }

   /** Proxy to setErrorHandler to THIS sub-terminal. */
   inline NonTerminal& operator << ( ErrorHandler hr )
   {
      return setErrorHandler(hr);
   }

   /** Proxy to set a name to THIS terminal, or start a subterminal. */
   inline NonTerminal& operator <<( const String& n ) {
      return addSubName(n);
   }

   /** Proxy to operator functors */
   inline NonTerminal& operator<<(NonTerminal& (*__pf)(NonTerminal&))
   {
      return __pf(*this);
   }


   /** Gets the count of sub-tokens. */
   virtual int arity() const;
   /** Gets the nth sub-token. */
   virtual Token* term( int ) const;
   /** Sets the nth sub-token. */
   virtual void term( int, Token* );
   /** Adds the next sub-token. */
   virtual void addTerm( Token* );
   
   void render( TextWriter& tw ) const;

   NonTerminal& addSubTerminal(Token& token);
   NonTerminal& addSubHandler(Handler hr);
   NonTerminal& addSubName(const String& name);

   /**
    * Starts a sub-ndonterminal definition.
    */
   static NonTerminal& sr(NonTerminal& nt);

   /**
    * Starts the next sub-ndonterminal definition.
    */
   static NonTerminal& nr(NonTerminal& nt);

   /**
    * Terminates a sub-nonterminal definition.
    */
   static NonTerminal& endr(NonTerminal& nt);

   /**
    * proxy Adds to current sub-terminal.
    */
   inline NonTerminal& operator <<(Token& token)
   {
      return addSubTerminal(token);
   }

   /** Proxy to set handler for current sub-terminal. */
   inline NonTerminal& operator << ( Handler hr )
   {
      return addSubHandler(hr);
   }

   class BuildError
   {
   public:
      BuildError(const NonTerminal& src, const String& descr);
      const String& descr() const { return m_descr; }
   private:
      String m_descr;
   };

   bool isDynamic() const { return m_isDynamic; }
   void setDynamic() { m_isDynamic = true; }

   Handler applyHandler() const { return m_handler; }

   /** Automatically set to true when adding itself as sub-token.
    *
    * This adds a small optimization in the compiler. When the compiler
    * matches a recursive token, instead of popping its context and
    * give control to the parent rule frame, it first gives the control back
    * to the same frame where the match was succesful.
    */
   bool isRecursive() const {return m_bIsRecursive; }

private:
   ErrorHandler m_eh;
   Handler m_handler;
   NonTerminal* m_currentSubNT;
   bool m_isDynamic;
   bool m_bIsRecursive;

   class Private;
   Private* _p;

   void subRender( TextWriter& tw, void* v ) const;
};

}
}

#endif	/* _FALCON_PARSER_NONTERMINAL_H_ */

/* end of parser/nonterminal.h */
