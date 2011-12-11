/*
   FALCON - The Falcon Programming Language.
   FILE: textwriter.h

   Falcon core module -- Wrapping for text writer
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CORE_TEXTWRITER_H
#define FALCON_CORE_TEXTWRITER_H


#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/types.h>
#include <falcon/textwriter.h>

#include <falcon/cm/stream.h>

namespace Falcon {

class Stream;
class ClassStream;

namespace Ext {

/** Holder for the stream. */
class FALCON_DYN_CLASS TextWriterCarrier: public UserCarrierT<StreamCarrier>
{
public:
   // we won't be separating the text readers and writers.
   TextWriter m_writer;
   
   TextWriterCarrier( StreamCarrier* stc );
   virtual ~TextWriterCarrier();    
   
   virtual void gcMark( uint32 mark );
};


/*# @class TextStream
   @param stream A stream on which to operate.
 
 */
class ClassTextWriter: public ClassUser
{
public:
   /** Create the textstream class.
    \param clsStream A ClassStream that will be known in the owning module.
    */
   ClassTextWriter( ClassStream* clsStream );
   virtual ~ClassTextWriter();

   //=============================================================
   //
   virtual void* createInstance( Item* params, int pcount ) const;
   
   
private:   
   ClassStream* m_clsStream;
   //====================================================
   // Properties.
   //
   FALCON_DECLARE_PROPERTY( encoding );
   FALCON_DECLARE_PROPERTY( crlf );
   FALCON_DECLARE_PROPERTY( lineflush );
   FALCON_DECLARE_PROPERTY( buffer );
   
   FALCON_DECLARE_METHOD( write, "text:S, count:[N], start:[N]" );
   FALCON_DECLARE_METHOD( writeLine, "text:S, count:[N], start:[N]" );   
   FALCON_DECLARE_METHOD( putChar, "char:S|N" );   
   FALCON_DECLARE_METHOD( getStream, "" );      
   FALCON_DECLARE_METHOD( flush, "" );
};

}
}

#endif	/* FALCON_CORE_TEXTWRITER_H */

/* end of textwriter.h */
