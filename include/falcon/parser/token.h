/*
   FALCON - The Falcon Programming Language.
   FILE: parser/token.h

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Apr 2011 17:16:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_TOKEN_H_
#define _FALCON_PARSER_TOKEN_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {
namespace Parsing {

class TokenInstance;

/** Generic token for Falcon generic parser.
 
 Tokens are the minimal unit of data which the lexer can recognize and produce.

 Tokens are associated with a symbolic name that indicates their significance
 and with an ID that is used in sequences to for tokens. IDs are actually
 hash valuse generated from token name, so that pattern matching doesn't
 require string comparison.

 Token subclasses may be associated with a value that they usually own. The virtual
 method detachValue() is meant so that subclasses can implement it to disengage
 deep values that are used (assembled) elsewhere.
 */
class FALCON_DYN_CLASS Token
{
public:
   typedef void(*deletor)(void*);
   virtual ~Token();

   const String& name() const { return m_name; }

   int prio() const { return m_prio; }
   void prio( int p ) { m_prio = p; }

   /** Checks if this token is nonterminal.
    \return true if the token is nonterminal.
    
    If the token is nonterminal it can be safely cast to NonTerminal.
    */
   bool isNT() const { return m_bNonTerminal; }

   /** Checks if this token is nonterminal.
    \return true if the token is nonterminal.

    If the token is nonterminal it can be safely cast to NonTerminal.
    */
   bool isRightAssoc() const { return m_bRightAssoc; }

   void setRightAssoc( bool bMode ) { m_bRightAssoc = bMode; }

   /** Creates a "match instance" of this token.
    \param v An actual value that this token has assumed during parsing.
    \return a newly created token instance.

    This method generates a token instance (read, actual parsed value) having this instance
    as its token, and the given value as the instance value.

    The value may be zero if the token is not meant to assume a parsed value;
    this is the case of literals.

    @see TokenInstance
    */
   TokenInstance* makeInstance( int line, int chr, void* data, deletor d );

   TokenInstance* makeInstance( int line, int chr, int32 v );
   TokenInstance* makeInstance( int line, int chr, uint32 v );
   TokenInstance* makeInstance( int line, int chr, int64 v );
   TokenInstance* makeInstance( int line, int chr, numeric v );
   TokenInstance* makeInstance( int line, int chr, bool v );
   TokenInstance* makeInstance( int line, int chr, const String& v );

   TokenInstance* makeInstance( int line, int chr );

   virtual int arity() const { return 0; }
   virtual Token* term( int ) const { return 0; }
   virtual void term( int, Token* ) {}
   virtual void addTerm( Token* ) {}

   bool isSimple() const { return m_bSimple; }
protected:
   bool m_bNonTerminal;
   bool m_bRightAssoc;
   bool m_bSimple;
   int m_prio;
   
   Token(const String& name, int prio = 0, bool bRightAssoc = false );
   Token();
   static uint32 simpleHash( const String& v );

   void name( const String& n );
private:
   String m_name;
};

}
}

#endif	/* _FALCON_PARSER_TOKEN_H_ */

/* end of parser/token.h */
