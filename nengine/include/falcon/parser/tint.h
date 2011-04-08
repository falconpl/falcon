/*
   FALCON - The Falcon Programming Language.
   FILE: parser/tint.h

   Token representing an integer numberfound by the lexer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_TINT_H_
#define	_FALCON_PARSER_TINT_H_

#include <falcon/setup.h>
#include <falcon/parser/terminal.h>

namespace Falcon {
namespace Parser {

/** Terminal token: integer number.
 */
class FALCON_DYN_CLASS TInt: public Terminal
{
public:
   inline TInt():
      Terminal("Int")
   {
   }

   inline virtual ~TInt() {}
};

/** Predefined integer token instance. */
extern FALCON_DYN_SYM TInt t_int;

}
}

#endif	/* _FALCON_PARSER_TINT_H_ */

/* end of parser/tint.h */
