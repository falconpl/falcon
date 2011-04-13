/*
   FALCON - The Falcon Programming Language.
   FILE: parser/tname.h

   Token representing a symbol found by the lexer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_TNAME_H_
#define	_FALCON_PARSER_TNAME_H_

#include <falcon/setup.h>
#include <falcon/parser/terminal.h>

namespace Falcon {
namespace Parsing {

/** Terminal token: name.

 Token representing symbols, variable names, etc. They hold a literal string.
 */
class FALCON_DYN_CLASS TName: public Terminal
{
public:
   inline TName():
      Terminal("Name")
   {
   }

   inline virtual ~TName() {}
};

/** Predefined string token instance. */
extern FALCON_DYN_SYM TName& t_name();

}
}

#endif	/* _FALCON_PARSER_TNAME_H_ */

/* end of parser/tname.h */
