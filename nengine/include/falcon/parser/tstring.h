/*
   FALCON - The Falcon Programming Language.
   FILE: parser/tstring.h

   Token representing a string found by the lexer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_TSTRING_H_
#define	_FALCON_PARSER_TSTRING_H_

#include <falcon/setup.h>
#include <falcon/parser/terminal.h>

namespace Falcon {
namespace Parser {

/** Terminal token: string.

 Strings are common entities in parse problems; this class is offered
 as a simple base representation of parsed strings as Falcon::String values.
 */
class FALCON_DYN_CLASS TString: public Terminal
{
public:
   inline TString():
      Terminal("String")
   {
   }
      
   inline virtual ~TString() {}
};

/** Predefined string token instance. */
extern FALCON_DYN_SYM TString t_string;

}
}

#endif	/* _FALCON_PARSER_TSTRING_H_ */

/* end of parser/tstring.h */
