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
#define	_FALCON_PARSER_TOKEN_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {
namespace Parser {

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

   uint32 id() const { return m_nID; }
   const String& name() const { return m_name; }

   /** Creates a "match instance" of this token.
    \param v An actual value that this token has assumed during parsing.
    \return a newly created token instance.

    This method generates a token instance (read, actual parsed value) having this instance
    as its token, and the given value as the instance value.

    The value may be zero if the token is not meant to assume a parsed value;
    this is the case of literals.

    @see TokenInstance
    */
   TokenInstance* makeInstance( void* data, deletor d );

   TokenInstance* makeInstance( int32 v );
   TokenInstance* makeInstance( uint32 v );
   TokenInstance* makeInstance( int64 v );
   TokenInstance* makeInstance( numeric v );
   TokenInstance* makeInstance( bool v );
   TokenInstance* makeInstance( const String& v );

protected:
   Token(uint32 nID, const String& name );
   Token(const String& name);

   static uint32 simpleHash( const String& v );

private:
   uint32 m_nID;
   String m_name;
};

}
}

#endif	/* _FALCON_PARSER_TOKEN_H_ */

/* end of parser/token.h */
