/*
   FALCON - The Falcon Programming Language.
   FILE: textreader.h

   Falcon core module -- Wrapping for text reader
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CORE_TEXTREADER_H
#define FALCON_CORE_TEXTREADER_H

#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/types.h>
#include <falcon/textreader.h>

#include <falcon/classes/classstream.h>

namespace Falcon {

class Stream;
class ClassStream;

namespace Ext {

/*# @class TextStream
   @param stream A stream on which to operate.
 
 */
class ClassTextReader: public ClassUser
{
public:
   /** Create the textstream class.
    \param clsStream A ClassStream that will be known in the owning module.
    */
   ClassTextReader( ClassStream* clsStream );
   virtual ~ClassTextReader();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   
private:   
   ClassStream* m_clsStream;
   
   //====================================================
   // Properties.
   //
   FALCON_DECLARE_PROPERTY( encoding );
   
   FALCON_DECLARE_METHOD( read, "text:S, count:N" );
   FALCON_DECLARE_METHOD( grab, "count:N" );
   FALCON_DECLARE_METHOD( readLine, "text:S, count:[N]" );
   FALCON_DECLARE_METHOD( grabLine, "count:[N]" );
   FALCON_DECLARE_METHOD( readEof, "text:S" );
   FALCON_DECLARE_METHOD( grabEof, "" );
   FALCON_DECLARE_METHOD( readRecord, "text:S, marker:S, count:[N]" );
   FALCON_DECLARE_METHOD( grabRecord, "marker:S, count:[N]" );
   FALCON_DECLARE_METHOD( readToken, "text:S, tokens:A, count:[N]" );
   FALCON_DECLARE_METHOD( grabToken, "tokens:A, count:[N]" );
   
   FALCON_DECLARE_METHOD( readChar, "text:S, append:[B]" );
   FALCON_DECLARE_METHOD( getChar, "" );
   FALCON_DECLARE_METHOD( ungetChar, "char:S|N" );   
   
   FALCON_DECLARE_METHOD( getStream, "" );      
   FALCON_DECLARE_METHOD( sync, "" );
   FALCON_DECLARE_METHOD( close, "" );

};

}
}

#endif	/* FALCON_CORE_TEXTREADER_H */

/* end of textreader.h */
