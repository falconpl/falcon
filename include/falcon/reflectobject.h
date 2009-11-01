/*
   FALCON - The Falcon Programming Language.
   FILE: reflectobject.h

   Base abstract class for object with reflective data (providing a user data).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 25 Jan 2009 15:42:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_REFLECT_OBJECT_H
#define FLC_REFLECT_OBJECT_H

#include <falcon/coreobject.h>

namespace Falcon
{

class FALCON_DYN_CLASS ReflectObject: public CoreObject
{
public:
   ReflectObject( const CoreClass* generator, void* userData );
   ReflectObject( const CoreClass* generator, FalconData* fdata );
   ReflectObject( const CoreClass* generator, Sequence* seq );
   
   ReflectObject( const ReflectObject &other );
   
   virtual ~ReflectObject();
   
   virtual bool setProperty( const String &prop, const Item &value );
   virtual bool getProperty( const String &key, Item &ret ) const;
   virtual ReflectObject *clone() const;
   
};

}

#endif

/* end of reflectobject.h */
