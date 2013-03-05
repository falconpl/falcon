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

#include <falcon/types.h>
#include <falcon/textreader.h>
#include <falcon/textwriter.h>

#include <falcon/classes/classstream.h>


namespace Falcon {

class Stream;

namespace Ext {


/** Holder for the stream and its TextReader/TextWriter helpers. */
class FALCON_DYN_CLASS TextStreamCarrier
{
public:
   // we won't be separating the text readers and writers.
   TextReader* m_reader;
   TextWriter* m_writer;
   String m_encoding;

   TextStreamCarrier( Stream* sc );
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
};

}
}

#endif	/* FALCON_CORE_TEXTSTREAM_H */

/* end of textstream.h */
