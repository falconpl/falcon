/*
   FALCON - The Falcon Programming Language.
   FILE: objectfactory.h

   Definition for the function used to create object instances.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 25 Jan 2009 13:09:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_OBJECT_FACTORY_H
#define FLC_OBJECT_FACTORY_H

#include <falcon/setup.h>

namespace Falcon
{

class CoreObject;
class CoreClass;

   /** Defines the core object factory function type.
      This factory function is used by core classes to create an object of the
      appropriate type.
      
      \param cls The class that is generating the givenn object.
      \param user_data A data that is to be stored in the object instance.
      \param deserializing true to prevent object structure initialization (it will be read
         from a somewhere else afterwards).
   */
   
   typedef CoreObject* (*ObjectFactory)( const CoreClass *cls, void *user_data, bool bDeserializing );

   /** Factory function creating a standard non-shelled object.
      If \b data is provided, it will be set as an opaque object in a standard non reflective
      FalconObject instance.
   */
   CoreObject* OpaqueObjectFactory( const CoreClass *cls, void *data, bool bDeserializing );
   
   /** Factory function creating a standard complete FalconObject.
      If \b data is provided, it will be cast to a FalconData (standard minimal carrier
      instance known by the system) and set in the returned FalconObject.
      
      \note This is the default factory type set for all the classes which doesn't provide 
      property level reflectivity.
   */
   CoreObject* FalconObjectFactory( const CoreClass *cls, void *data, bool bDeserializing );
   
   /** Factory function creating a standard complete FalconObject and holding a sequence.
      If \b data is provided, it will be cast to a Sequence (FalconData with sequential access
      facilities) and set in the returned FalconObject.
   */
   CoreObject* FalconSequenceFactory( const CoreClass *cls, void *data, bool bDeserializing );
   
   /** Factory function creating a full reflective non-shelled object.
      
      If \b data is provided, it will be set as an opaque object in a fully reflective class.
      
      All the properties in the generator class must provide reflectivity. The user_data will
      be considered opaque and will not take part in object-wide actions (GC marking, destruction,
      cloning and so on).
   */
   CoreObject* ReflectOpaqueFactory( const CoreClass *cls, void *user_data, bool );

   /** Factory function creating a full reflective FalconData object.
      
      If \b data is provided, it will be set as a FalconData object in a fully reflective class.
      
      Being a FalconData, the user data will participate in the life of the reflector object,
      i.e. being cloned, GC marked or destroyed with its owner.
      
      \note This is the default factory functions for classes presenting all their properties
         as reflective.
   */
   CoreObject* ReflectFalconFactory( const CoreClass *cls, void *user_data, bool );
   
   /** Factory function creating a full reflective Sequence object.
      
      If \b data is provided, it will be set as a Sequence object in a fully reflective class.
      
      Being a derived from a FalconData, the user data will participate in the life of the reflector object,
      i.e. being cloned, GC marked or destroyed with its owner.
   */
   CoreObject* ReflectSequenceFactory( const CoreClass *cls, void *user_data, bool );

   /** Factory function creating a partially reflective non-shelled object.
      
      If \b data is provided, it will be set as an opaque object in a fully reflective class.
      
      The user_data will
      be considered opaque and will not take part in object-wide actions (GC marking, destruction,
      cloning and so on).
   */
   CoreObject* CROpaqueFactory( const CoreClass *cls, void *user_data, bool bDeserial );

   /** Factory function creating a partially reflective FalconData object.
      
      If \b data is provided, it will be set as a FalconData object in a fully reflective class.
      
      Being a FalconData, the user data will participate in the life of the reflector object,
      i.e. being cloned, GC marked or destroyed with its owner.
      
      \note This is the default factory functions for classes presenting some of their properties
         as reflective.
   */
   CoreObject* CRFalconFactory( const CoreClass *cls, void *user_data, bool bDeserial );
   
   /** Factory function creating a partially reflective Sequence object.
      
      If \b data is provided, it will be set as a Sequence object in a partially reflective class.
      
      Being a derived from a FalconData, the user data will participate in the life of the reflector object,
      i.e. being cloned, GC marked or destroyed with its owner.
   */
   CoreObject* CRSequenceFactory( const CoreClass *cls, void *user_data, bool bDeserial );
}
#endif

/* end of objectfactory.h */
