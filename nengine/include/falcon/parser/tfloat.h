/*
   FALCON - The Falcon Programming Language.
   FILE: parser/tfloat.h

   Token representing a floating point number found by the lexer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_TFLOAT_H_
#define	_FALCON_PARSER_TFLOAT_H_

#include <falcon/setup.h>
#include <falcon/parser/terminal.h>

namespace Falcon {
namespace Parser {

/** Terminal token: floating point number.
 */
class FALCON_DYN_CLASS TFloat: public Terminal
{
public:
   inline TFloat():
      Terminal("Float")
   {
   }

   inline virtual ~TFloat() {}
};

/** Predefined string token instance. */
extern FALCON_DYN_SYM TFloat t_float;

}
}

#endif	/* _FALCON_PARSER_TFLOAT_H_ */

/* end of parser/tfloat.h */
