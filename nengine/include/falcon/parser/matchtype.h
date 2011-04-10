/*
   FALCON - The Falcon Programming Language.
   FILE: parser/matchtype.h

   Enumeration used to propagate match status.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 16:44:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_PARSER_MATCHTYPE_H
#define	_FALCON_PARSER_MATCHTYPE_H

namespace Falcon {
namespace Parsing {

/** Enumeration representing the match status of a rule. */
typedef enum {
   /** Up to date, matching, but still not able to decide. */
   t_tooShort,
   /** Match failed. */
   t_nomatch,
   /** Match success. */
   t_match
} t_matchType;

}
}

#endif	/* _FALCON_PARSER_MATCHTYPE_H */

/* end of parser/matchtype.h */
