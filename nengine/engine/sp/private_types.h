/*
   FALCON - The Falcon Programming Language.
   FILE: private_types.h

   Falcon source parser -- Types that are private to the sourceparser
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Jul 2011 17:34:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SP_PRIVATE_TYPES_H_
#define	_FALCON_SP_PRIVATE_TYPES_H_

#include <falcon/string.h>
#include <deque>

namespace Falcon {

class Expression;

typedef std::deque<Expression*> List;
typedef std::deque<String> NameList;
class PairList: public std::deque< std::pair<Expression*,Expression*> >
{
public:
   PairList():
      m_bHasPairs(false)
   {}

   bool m_bHasPairs;
};

}

#endif	/* _FALCON_SP_PRIVATE_TYPES_H_ */

/* end of private_types.h */
