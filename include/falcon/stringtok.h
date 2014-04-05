/*
   FALCON - The Falcon Programming Language.
   FILE: stringtok.h

   String tokenizer general purpose class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Apr 2014 15:01:12 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STRINGTOK_H_
#define _FALCON_STRINGTOK_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon {

/** String tokenizer general purpose class.
 *
 */
class FALCON_DYN_CLASS StringTokenizer
{
public:

   StringTokenizer( const String& source, const String& token, bool bGroupTokens=false )

   bool next( String& target );
   bool hasNext() const;

private:
   String m_source;
   String m_token;

   uint32 m_chr;
   uint32 m_pos;

   bool m_bGroupTokens;
   bool m_bIsChr;

   void nextChr( String& target );
   void nextStr( String& target );
};

}

#endif

/* end of stringtok.h */
