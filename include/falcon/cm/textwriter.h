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

#include <falcon/class.h>
#include <falcon/types.h>
#include <falcon/textwriter.h>

namespace Falcon {

class Stream;
class ClassStream;

namespace Ext {

/*# @class TextStream
   @param stream A stream on which to operate.
 
 */
class ClassTextWriter: public Class
{
public:
   /** Create the textstream class.
    \param clsStream A ClassStream that will be known in the owning module.
    */
   ClassTextWriter( ClassStream* clsStream );
   virtual ~ClassTextWriter();
   
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

#endif	/* FALCON_CORE_TEXTWRITER_H */

/* end of textwriter.h */
