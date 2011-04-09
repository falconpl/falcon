/*
   FALCON - The Falcon Programming Language.
   FILE: parser/eol.h

   Token representing an End of Line found by lexer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_TEOL_H_
#define	_FALCON_PARSER_TEOL_H_

#include <falcon/setup.h>
#include <falcon/parser/terminal.h>

namespace Falcon {
namespace Parser {

/** Terminal token: end of line.

 Token representing symbols, variable names, etc. They hold a literal string.
 */
class FALCON_DYN_CLASS TEol: public Terminal
{
public:
   inline TEol():
      Terminal("EOL")
   {
   }

   inline virtual ~TEol() {}
};

/** Predefined end of line token instance. */
extern FALCON_DYN_SYM TEol t_eol;

}
}

#endif	/* _FALCON_PARSER_TEOL_H_ */

/* end of parser/teol.h */
