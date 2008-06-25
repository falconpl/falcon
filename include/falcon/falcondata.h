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
#include <falcon/objectmanager.h>

namespace Falcon {

/** Common falcon inner object data infrastructure */

class FALCON_DYN_CLASS FalconData: public BaseAlloc
{
public:
   virtual ~FalconData() {}

   virtual bool isSequence() const { return false; }

   virtual void gcMark( VMachine *mp ) = 0;

   virtual FalconData *clone() const = 0;
};


/** Object manager for Falcon data.
   The FalconData class infrastructure has a common interface
   which can be used to fulfil the needs of a standard object
   manager.

   Instead of creating many object managers, we provide just
   one for all the class derived from FalconData, and let
   their virtual methods to handle the events that the
   data manager receives.

   Applications and modules willing to provide object user_data
   which respects FalconData interface may use this manager
   instead of providing their own.

   It is possible to provide a dummy instance of the FlaconData
   subclass that should be handled by the CoreClass using this
   FalconDataManager at construction; in that case, the dummy
   data will be used as a "stamp" to create an initial user_data
   via the FalconData::clone() method at init.
*/

class FalconDataManager: public ObjectManager
{
   FalconData *m_model;

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
   FalconDataManager( FalconData *model=0 ):
      m_model( model )
   {}

   virtual ~FalconDataManager();

   virtual bool isFalconData() const { return true; }

   /** Initializes the user data in the object as a FalconData.

      If a model has been provided to the constructor of this class,
      that model is used and cloned here. The result is set as
      user_data in the given object.

      If no stamp were given, this method does nothing.
   */

   virtual void *onInit( VMachine *vm );

   virtual void onGarbageMark( VMachine *vm, void *data );

   virtual void onDestroy( VMachine *vm, void *user_data );

   virtual void *onClone( VMachine *vm, void *user_data );
};


extern FALCON_DYN_SYM FalconDataManager core_falcon_data_manager;

}

#endif

/* end of falcondata.h */
