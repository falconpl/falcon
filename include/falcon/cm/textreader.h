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

#include <falcon/class.h>
#include <falcon/types.h>
#include <falcon/textreader.h>

#include <falcon/classes/classstream.h>

namespace Falcon {

class Stream;
class ClassStream;

namespace Ext {

/*# @class TextStream
   @param stream A stream on which to operate.

   @prop underlying The underlying stream
   @prop eof True when all the data from the stream has been read.
 
 */
class ClassTextReader: public Class
{
public:
   /** Create the textstream class.
    \param clsStream A ClassStream that will be known in the owning module.
    */
   ClassTextReader( ClassStream* clsStream );
   virtual ~ClassTextReader();

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   
private:   
   ClassStream* m_clsStream;
};

}
}

#endif	/* FALCON_CORE_TEXTREADER_H */

/* end of textreader.h */
