/*
   FALCON - The Falcon Programming Language.
   FILE: streamtok.h

   Advanced general purpose stream tokenizer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 23 Apr 2014 10:45:42 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STREAMTOK_H_
#define _FALCON_STREAMTOK_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace re2 {
   class RE2;
}

namespace Falcon {

class TextReader;

/** Advanced general purpose stream tokenizer.
 *
 *  This class breaks a stream of data into tokenized substrings via iterative calls
 *  to the next() method, and via callbacks that allow to associate specific actions
 *  with the tokens being found.
 *
 *  Multiple token and regular expressions tokens are supported.
 *
 *  The tokenizer scope of application is limited by a buffer size, which is the maximum
 *  size of the token splits that can be individuated.
 *
 */
class FALCON_DYN_CLASS StreamTokenizer
{
public:

   static const uint32 DEFAULT_BUFFER_SIZE = 4096;

   /** Create a tokenizer.
    */
   StreamTokenizer( TextReader* source, uint32 bufsize=DEFAULT_BUFFER_SIZE );
   StreamTokenizer( const String& source );
   StreamTokenizer( const StreamTokenizer& other );

   virtual ~StreamTokenizer();

   void setSource(TextReader* source, uint32 bufsize=DEFAULT_BUFFER_SIZE );
   void setSource(const String& source );

   bool next();
   bool hasNext() const;

   uint32 currentMark() const;
   void gcMark( uint32 mark );

   virtual void onTokenFound(  int32 id, String* textContent, String* tokenContent, void* userData );

   void setOwnText( bool mode ) { m_bOwnText= mode; }
   bool ownText() const { return m_bOwnText; }
   void setOwnToken( bool mode ) { m_bOwnToken = mode; }
   bool ownToken() const { return m_bOwnToken; }

   typedef void (*deletor)(void*);

   void addToken( const String& token, void *data = 0, deletor del = 0 );
   void addRE( const String& re, void *data = 0, deletor del = 0  );
   void addRE( re2::RE2* regex, void* data = 0, deletor del = 0 );
   void onResidual(void* data, deletor del = 0);

   void rewind();

private:
   TextReader* m_tr;

   uint32 m_bufSize;
   uint32 m_bufPos;
   uint32 m_bufLen;
   char* m_buffer;

   String* m_runningText;
   String* m_runningToken;

   uint32 m_mark;
   bool m_bOwnToken;
   bool m_bOwnText;

   // set to true when there is some regex in tokens.
   bool m_hasRegex;

   class Private;
   Private* _p;

   void init();
   bool refill();
   bool findNext( int32& id, length_t& pos);
   length_t findText( const char* token, length_t len );
};

}

#endif

/* end of streamtok.h */
