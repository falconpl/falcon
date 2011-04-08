/*
   FALCON - The Falcon Programming Language.
   FILE: parser/literal.h

   Token representing a literal grammar terminal.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_LITERAL_H_
#define	_FALCON_PARSER_LITERAL_H_

#include <falcon/setup.h>
#include <falcon/parser/terminal.h>

namespace Falcon {
namespace Parser {

/** Literal grammar terminal symbol.
 This token matches exaclty one instance of a text string in input.

 Example:
 
 @code
 ...
 Literal TPlus( "+" );
 Literal TMinus( "-" );

 Rule RAdd = MakeRule("Add", AddEffector ).r(Number).r(TPlus).r(Number);
 Rule RSub = MakeRule("Sub", SubEffector ).r(Number).r(TMinus).r(Number);
 ...
 @endcode
 */
class FALCON_DYN_CLASS Literal: public Terminal
{
public:
   /** Creates a literal token with a distinct name.*/
   inline Literal( const String& name, const String& token ):
      Terminal( name ),
      m_token( token )
      {}

   /** Creates a literal token using the token string as the literal name.*/
   inline Literal( const String& token ):
      Terminal( token ),
      m_token( token )
      {}

   virtual ~Literal();

   /** Returns the literal token associated with this instance. */
   inline const String& token() const { return m_token; }

private:
   String m_token;
};

}
}

#endif	/* _FALCON_PARSER_LITERAL_H_ */

/* end of parser/literal.h */
