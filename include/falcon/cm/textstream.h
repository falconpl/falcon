/*
   FALCON - The Falcon Programming Language.
   FILE: textstream.h

   Falcon core module -- Abstraction integrating text streams
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_TEXTSTREAM_H
#define FALCON_CORE_TEXTSTREAM_H

#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/types.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>

#include <falcon/cm/stream.h>

namespace Falcon {

class Stream;

namespace Ext {


/** Holder for the stream and its TextReader/TextWriter helpers. */
class FALCON_DYN_CLASS TextStreamCarrier: public StreamCarrier
{
public:
   // we won't be separating the text readers and writers.
   TextReader m_reader;
   TextWriter m_writer;
   String m_encoding;

   
   TextStreamCarrier( Stream* stream );
   virtual ~TextStreamCarrier();   
   virtual void onFlushingOperation();   
   
   bool setEncoding( const String& encName );
};


/*# @class TextStream
   @param stream A stream on which to operate.
 
 */
class ClassTextStream: public ClassStream
{
public:
   /** Create the textstream class.
    \param parent A ClassStream that will be known in the owning module
    as the parent of this class.
    */
   ClassTextStream( ClassStream* parent );
   virtual ~ClassTextStream();
   virtual void* createInstance() const;
   
   //=============================================================
   //
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   
private:   
   
   // keeping a reference for simplicity
   ClassStream* m_stream;
   //====================================================
   // Properties.
   //
   FALCON_DECLARE_PROPERTY( encoding );
   
   FALCON_DECLARE_METHOD( write, "text:S, count:[N], start:[N]" );
   FALCON_DECLARE_METHOD( read, "text:S, count:[N]" );   
   FALCON_DECLARE_METHOD( grab, "count:N" );
   FALCON_DECLARE_METHOD( readLine, "text:S, maxCount:[N]" );   
   FALCON_DECLARE_METHOD( grabLine, "maxCount:[N]" );   
   FALCON_DECLARE_METHOD( readChar, "text:S, append:[B]" );
   FALCON_DECLARE_METHOD( getChar, "" );
   FALCON_DECLARE_METHOD( ungetChar, "char:S|N" );
   FALCON_DECLARE_METHOD( putChar, "char:S|N" );
   
};

}
}

#endif	/* FALCON_CORE_TEXTSTREAM_H */

/* end of textstream.h */
