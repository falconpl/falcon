/*
   FALCON - The Falcon Programming Language.
   FILE: UserData.h

   Falcon user simplified reflection architecture.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 22 Jun 2008 11:09:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon user simplified reflection architecture.
*/

#ifndef FALCON_USER_DATA_H
#define FALCON_USER_DATA_H

#include <falcon/setup.h>
#include <falcon/memory.h>
#include <falcon/falcondata.h>

namespace Falcon {

/** Common falcon inner object data infrastructure for simple reflective applications.

   This simplified reflection scheme allows to declare two methods in the data
   carried by falcon objects which perform reflection from and to the final
   object.

   UserData can be used both to carry directly needed data or to provide a "shell"
   managed by falcon to carry around data that is reflected through their interface.

   UserData interface is a relatively simple, but inefficient, way to reflect program
   data. Direct reflection through the programming of a specific ObjectManager for the
   reflected object is preferrable when performance are at stake.

   \note Remember to overload the base class method clone() (it is ok to return always
         zero if the semantic of the class doesn't allow cloning).
*/

class FALCON_DYN_CLASS UserData: public FalconData
{
public:
   virtual ~UserData() {}

   virtual void getProperty( CoreObject *obj, const String &propName, Item &prop ) = 0;
   virtual void setProperty( CoreObject *obj, const String &propName, const Item &prop ) = 0;

   /** Defaulting GcMarking to none. */
   virtual void gcMark( VMachine *mp ) {}
};


/** Object manager for User data.
   This object manager can be given to classes that will handle user data.
*/

class UserDataManager: public FalconDataManager
{
   bool m_bNeedCache;

public:
   /** Creates the instance, eventually providing a stamp model.
      If a stamp model is given, the onInit() method will use it's
      clone() virtual method to create a new instance of the needed
      classes. To configure it it's then necessary to have the
      object "_init" method called, or to configure it by hand.

      If the model is left to 0, onInit does nothing.

      The model is destroyed (via virtual destructor) when the manager
      is destroyed.
   */
   UserDataManager( bool bNeedCache=true, UserData *model=0 );
   virtual ~UserDataManager();

   virtual bool needCacheData() const { return m_bNeedCache; }
   virtual bool hasClassReflection() const { return true; }

   virtual bool onSetProperty( CoreObject *owner, void *user_data, const String &propname, const Item &property );
   virtual bool onGetProperty( CoreObject *owner, void *user_data, const String &propname, Item &property );
};

extern FALCON_DYN_SYM UserDataManager core_user_data_manager_cacheful;
extern FALCON_DYN_SYM UserDataManager core_user_data_manager_cacheless;

}

#endif

/* end of UserData.h */
