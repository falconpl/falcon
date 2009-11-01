/*
   FALCON - The Falcon Programming Language.
   FILE: crobject.h

   Falcon Cached/Reflective Object - Automatic reflected objects with
   a cache.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 24 Jan 2009 13:48:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon Object - Standard instance of classes in script
*/

#ifndef FLC_CR_OBJECT_H
#define FLC_CR_OBJECT_H

#include <falcon/setup.h>
#include <falcon/cacheobject.h>

namespace Falcon
{

class VMachine;
class Stream;

/** Reflective Object with cache.
   
   Classes that need to provide an item cache AND require reflection
   of some or all the properties must derive from this class.
*/
class FALCON_DYN_CLASS CRObject: public CacheObject
{

public:
   CRObject( const CoreClass* generator, bool bSeralizing = false );
   CRObject( const CRObject &other );
   virtual ~CRObject() {}

   virtual CRObject *clone() const;
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &key, Item &ret ) const;
};


}

#endif

/* end of crobject.h */
