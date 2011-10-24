/*
   FALCON - The Falcon Programming Language.
   FILE: deserializer.h

   Helper for cyclic joint structure deserialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 18 Oct 2011 17:45:15 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DESERIALIZER_H
#define FALCON_DESERIALIZER_H

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class Class;
class DataReader;
class ModSpace;

/** Helper for cyclic joint structure deserialization.
*/

class FALCON_DYN_CLASS Deserializer
{
public:
   Deserializer(ModSpace* srcSpace=0);
   virtual ~Deserializer();
   
   virtual void restore( DataReader* rd );
   
   virtual void* next( Class&* handler );
   virtual bool hasNext() const;
   virtual uint32 objCount() const;
   
private:      
   ModSpace* m_modSpace;
   
   class MetaData;
   MetaData* _meta;
   
   DataReader* m_rd;
   
   // Using void* because we'll be using private data for that.
   // void postDeserialize( MetaData& prv );   
};

}

#endif

/* end of deserializer.h */

