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
 *  This class breaks a string into tokenized substrings via iterative calls
 *  to the next() method.
 *
 *  This class doesn't support breaking of strings using multiple tokens,
 *  but it supports single-character or multi-character tokens. It also supports
 *  token grouping.
 */
class FALCON_DYN_CLASS StringTokenizer
{
public:

   /** Create a tokenizer using a token string.
    * \param source The string to be parsed.
    * \param token The token used for parsing.
    * \param bGroupTokens If set to true, adjiacent tokens will be trimmed.
    * \throw ParamError if the token is empty.
    *
    * If the source string is empty, then hasNext() method will return false
    * even before the first call to next().
    *
    * If the token is a string containing a single character, the single-character version
    * of the checks will be used (making the token matching more efficient).
    */
   StringTokenizer( const String& source, const String& token, bool bGroupTokens=false );

   /** Create a tokenizer using a single token character.
    * \param source The string to be parsed.
    * \param chr The token used for parsing.
    * \param bGroupTokens If set to true, adjiacent tokens will be trimmed.
    *
    * If the source string is empty, then hasNext() method will return false even before the first call to next().
    */
   StringTokenizer( const String& source, uint32 chr, bool bGroupTokens=false );
   StringTokenizer( const StringTokenizer& other );
   ~StringTokenizer() {}

   bool next( String& target );
   bool hasNext() const;

   uint32 currentMark() const { return m_source.currentMark(); }
   void gcMark( uint32 mark ) { m_source.gcMark(mark); }

   void setLimit(uint32 count) { m_limit = count; }
   void resetLimit() { m_limit = 0xffffffff; }
   uint32 count() const { return m_count; }

private:
   String m_source;
   String m_token;

   uint32 m_chr;
   uint32 m_pos;
   uint32 m_count;
   uint32 m_limit;

   bool m_bGroupTokens;
   bool m_bIsChr;

   void nextChr( String& target );
   void nextStr( String& target );
};

}

#endif

/* end of stringtok.h */
