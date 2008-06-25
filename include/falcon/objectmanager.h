/*
   FALCON - The Falcon Programming Language.
   FILE: objectmanager.h

   Base abstract class to manage inner Falcon object data.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 20 Jun 2008 19:49:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Base abstract class to manage inner Falcon object data.
*/

#ifndef FAL_OBJECT_MANAGER_H
#define FAL_OBJECT_MANAGER_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/cclass.h>

namespace Falcon {

class CoreObject;

/** Base abstract class to manage inner Falcon object data.
   Object manager is a function vector used to manipulate the
   user data member of Falcon Objects.

   It may be useful to cache Falcon GC sensible data
   internally in the user data or in the object hosting the data.

   If a class is "fully reflexive", that is, if all of its
   properties are reflexive or read-only, then the object
   instance is NOT given a property array. The user data may store
   internally garbage data and mark it when the GC callbacks are
   invoked.

   When a class is not fully reflective, then the property handled
   to reflection functions is not a transient data, but it is actually
   stored on the object structure. In example, it may be possible to
   check if the property incoming in reflection functions is valid
   and leave it untouched, or change it minimally instead of creating
   it anew.

   The manager has not "global reflection functions". Instead, each
   property may be reflected either directly through offset-from-start
   and conversion-size, which turns C data automatically into
   Falcon data, or through reflection functions that are called when
   the property is set or read.

   However, the ObjectManager can tell the engine two things:
   - If there is the need to create cache item data (even if the class is
      fully reflective)
   - If the reflection is immediate or deferred.

   With deferred reflection, the reflection takes place only when directly
   invoked by the CoreObject::reflectFrom and CoreObject::reflectInto.

   A very important object manager is the Property Manager, which
   manipulates standard properties in property vector.
*/

class FALCON_DYN_CLASS ObjectManager: public BaseAlloc
{
public:
   virtual ~ObjectManager() {}

   /** Indicates that reflection should not be conidered immediately.

      If this method returns true, (also, if the class is not given
      any ObjectManager, or if the object has not a user_data), the
      reflection informations stored in the class property table are
      not used during the normal lifetime of the object.

      The object must be manually reflected from external user_data
      through CoreObject::reflectFrom and CoreObject::reflectTo.
   */
   virtual bool isDeferred() const { return false; }

   /** Return true if the user data of this can be casted into a FalconData.
      FalconData is a common layer implemented for some special, engine-relevant
      user data that travel in the CoreObjects.
   */

   virtual bool isFalconData() const { return false; }

   /** Return true if the user data is to be managed through class reflection.
      If this method returns true, the methods onSetProperty and onGetProperty
      will be called when a property is get or set in an object implementing
      this ObjectManager.

      Single per-property reflection settings get the priority.

      Unless needCacheData returns true, this setting instructs the class
      manager that no cache data is needed for the instances of that classes;
      all caching should be local.
   */
   virtual bool hasClassReflection() const { return false; }

   /** Callback on property set on the managed object.

      This callback gets called if hasClassReflection() returns true,
      declaring that the class using this manager is willing to manage
      its own properties.

      Single per-property reflection settings get the priority, so the
      properties must be declared normally (it is not possible to have synthetic
      properties this way).
   */
   virtual void onSetProperty( CoreObject *owner, void *user_data, const String &propname, const Item &property ) {}

   /** Callback on property get on the managed object.

      This callback gets called if hasClassReflection() returns true,
      declaring that the class using this manager is willing to manage
      its own properties.

      Single per-property reflection settings get the priority, so the
      properties must be declared normally (it is not possible to have synthetic
      properties this way).
   */
   virtual void onGetProperty( CoreObject *owner, void *user_data, const String &propname, Item &property ) {}

   /** Tells the engine if this class should create cache Items.

      If this method returns true, the class instances are always filled with
      vectors of items, and the results of application of reflection is
      cached there.

      By default it returns false;
   */
   virtual bool needCacheData() const { return false; }

   /** Method called on object init.
      This method has the chance to initialize the owner and
      create its own user data.

      VM is reachable through the object.

      This method is free to call any method of the object. I.e. if
      it is known that other parts of this object have been setup,
      it may call a setProperty on a property provided by a subclass.

      If, for some reason, the manager doesn't want to initialize the
      user data in this moment, it may delegate that to the init method
      being called afterwards, or to other code being called after
      CoreClass::createInstance. The manager may also create the user_data
      and leave proper initialization to one of those steps.

      \param vm the VM where tha action takes place.
      \return the user data that will be added to this object
   */
   virtual void *onInit( VMachine *vm ) = 0;

   /** Called during the mark step of the VM.
      The object should ask the VM to mark all the items that are subject to garbage collection in the user data.

      By default, this method returns without doing nothing.
      \param vm The vm where the action takes place.
      \param user_data The data to be marked.
      \param mark Current garbage mark.

   */
   virtual void onGarbageMark( VMachine *vm, void *data ) {}

   /** Called during item destruction.
      \param vm The vm where the action takes place.
      \param user_data The data to be destroyed.
   */
   virtual void onDestroy( VMachine *vm, void *user_data ) = 0;

   /** Called during item clone.

      The method may return the same user data with an internal reference count
      incremented, a new instance of the data or 0 if the semantic of the user data
      makes it impossible to be cloned.

      In case of 0 returned, the VM may raise a CloneError.
      \param vm The vm where the action takes place.
      \param user_data The data to be cloned.
      \return 0 or a new instance (or referenced instance) of the user data.
   */
   virtual void *onClone( VMachine *vm, void *user_data ) = 0;

   /** Allows class-wide on-demand reflection.
      For some classes, reflection may be done once every while, on request.

      This mehtod is called by CoreObject::reflectTo(). If it returns true,
      it means that the ObjectManager has accepted the reflection request and
      has provided its own reflection. If it returns false, the object falls back
      to the standard per-property reflection mechanism.

      The base class implementation returns false.
   */
   virtual bool onObjectReflectTo( CoreObject *reflector, void *user_data ) { return false; }

   /** Allows class-wide on-demand reflection.
      For some classes, reflection may be done once every while, on request.

      This mehtod is called by CoreObject::reflectFrom(). If it returns true,
      it means that the ObjectManager has accepted the reflection request and
      has provided its own reflection. If it returns false, the object falls back
      to the standard per-property reflection mechanism.

      The base class implementation returns false.
   */
   virtual bool onObjectReflectFrom( CoreObject *reflector, void *user_data ) { return false; }

};

}

#endif

/* end of objectmanager.h */
