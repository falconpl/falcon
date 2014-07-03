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
   StreamTokenizer( const String& source, uint32 bufsize=DEFAULT_BUFFER_SIZE );
   StreamTokenizer( const StreamTokenizer& other );

   virtual ~StreamTokenizer();

   void setSource(TextReader* source );
   void setSource(const String& source );

   bool next();
   bool hasNext() const;

   uint32 currentMark() const;
   void gcMark( uint32 mark );

   virtual void onTokenFound(  int32 id, String* tokenContent, void* tokData ) = 0;
   virtual void onTextFound(  String* textContent, void* textData ) = 0;

   void setOwnText( bool mode );
   bool ownText() const;
   void setOwnToken( bool mode );
   bool ownToken() const;

   typedef void (*deletor)(void*);

   void addToken( const String& token, void *data = 0, deletor del = 0 );
   void addRE( const String& re, void *data = 0, deletor del = 0  );
   void addRE( re2::RE2* regex, void* data = 0, deletor del = 0 );
   void setTextCallbackData(void* data, deletor del = 0);
   void setTokenCallbackData(void* data, deletor del = 0);

   void* getTextCallbackData() const;
   void* getTokenCallbackData() const;

   void rewind();

   bool giveTokens() const;
   bool groupTokens() const;

   void giveTokens(bool b);
   void groupTokens(bool b);


   void clearTokens();

    //Need to do something about this
    String* m_runningText;
    //Need to do something about this
    String* m_runningToken;

private:
   TextReader* m_tr;
   String m_source;
   length_t m_srcPos;

   uint32 m_bufSize;
   uint32 m_bufLen;
   char* m_buffer;

   uint32 m_mark;
   bool m_bOwnToken;
   bool m_bOwnText;

   // set to true when there is some regex in tokens.
   bool m_hasRegex;
   bool m_hasLastToken;
   bool m_hasToken;
   bool m_tokenLoaded;

   bool m_giveTokens;
   bool m_groupTokens;
   String m_lastToken;
   int32 m_tokenID;

   uint32 m_maxTokenSize;

   class Private;
   Private* _p;

   void init();
   void reinit();
   bool findNext( int32& id, length_t& pos);
   length_t findText( const char* token, length_t len );

   bool getNextChar( char_t &chr );
};

}

#endif

/* end of streamtok.h */
