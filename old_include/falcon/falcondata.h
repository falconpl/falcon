/*
   FALCON - The Falcon Programming Language.
   FILE: falcondata.h

   Falcon common object reflection architecture.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jun 2008 11:09:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon common object reflection architecture.
*/

#ifndef FALCON_DATA_H
#define FALCON_DATA_H

#include <falcon/setup.h>
#include <falcon/memory.h>
#include <falcon/basealloc.h>

namespace Falcon {

class MemPool;
class Stream;

/** Common falcon inner object data infrastructure */

class FALCON_DYN_CLASS FalconData: public BaseAlloc
{
public:
   virtual ~FalconData() {}

   virtual bool isSequence() const { return false; }

   virtual void gcMark( uint32 mark ) = 0;

   virtual FalconData *clone() const = 0;
   
   /** Serializes this instance on a stream.
      \throw IOError in case of stream error.
   */
   virtual bool serialize( Stream *stream, bool bLive ) const;
   
   /** Deserializes the object from a stream.
      The object should be created shortly before this call, giving 
      instruction to the constructor not to perform a full initialization,
      as the content of the object will be soon overwritten.
      
      Will throw in case of error.
      \throw IOError in case of stream error.
      \param stream The stream from which to read the object.
      \param bLive If true, 
      \return External call indicator. In case it returns true, the caller
         should 
   */
   virtual bool deserialize( Stream *stream, bool bLive );
};

}

#endif

/* end of falcondata.h */
