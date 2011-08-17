/*
   FALCON - The Falcon Programming Language.
   FILE: falconobject.h

   Falcon Object - Standard instance of classes in script
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

#ifndef FLC_FALCON_OBJECT_H
#define FLC_FALCON_OBJECT_H

#include <falcon/setup.h>
#include <falcon/cacheobject.h>

namespace Falcon
{

class VMachine;
class FalconData;

class FALCON_DYN_CLASS FalconObject: public CacheObject
{
public:
   FalconObject( const CoreClass* generator, bool bSeralizing = false );

   FalconObject( const FalconObject &other );

   virtual ~FalconObject();

   virtual FalconObject *clone() const;
};

}

#endif

/* end of falconobject.h */
