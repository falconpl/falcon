/*
   FALCON - The Falcon Programming Language.
   FILE: classstream.h

   Falcon core module -- Interface to Stream.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSSTREAM_H
#define FALCON_CLASSSTREAM_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/class.h>
#include <falcon/types.h>
#include <falcon/refcounter.h>
#include <falcon/path.h>

namespace Falcon {

class Stream;
class StreamBuffer;
class Transcoder;

/*# @class Stream
   
 */
class FALCON_DYN_CLASS ClassStream: public Class
{
public:
   
   ClassStream();
   ClassStream(const String& subclassName );
   virtual ~ClassStream();

   //=============================================================
   // Using a different carrier.
   
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void* createInstance() const;
   
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   
   Selectable* getSelectableInterface( void* instance ) const;
};

}

#endif

/* end of classstream.h */
