/*
   FALCON - The Falcon Programming Language.
   FILE: parser/teof.h

   Token representing a the end of file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 10 Apr 2011 11:28:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_EOF_H_
#define	_FALCON_PARSER_EOF_H_

#include <falcon/setup.h>
#include <falcon/parser/terminal.h>

namespace Falcon {
namespace Parsing {

/** Terminal token: end of file.
 */
class FALCON_DYN_CLASS TEof: public Terminal
{
public:
   inline TEof():
      Terminal("Float")
   {
   }

   inline virtual ~TEof() {}
};

/** Predefined string token instance. */
extern FALCON_DYN_SYM TEof t_eof;

}
}

#endif	/* _FALCON_PARSER_EOF_H_ */

/* end of parser/teof.h */
