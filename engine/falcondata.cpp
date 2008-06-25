/*
   FALCON - The Falcon Programming Language.
   FILE: falcondata.cpp

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

#include <falcon/falcondata.h>

namespace Falcon {


FalconDataManager::~FalconDataManager() {}

void *FalconDataManager::onInit( VMachine *vm )
{
   if ( m_model != 0 )
      return m_model->clone();
   return 0;
}

void FalconDataManager::onGarbageMark( VMachine *vm, void *data )
{
   FalconData *fd = static_cast<FalconData*>(data);
   fd->gcMark( vm );
}

void FalconDataManager::onDestroy( VMachine *vm, void *user_data )
{
      FalconData *fd = static_cast<FalconData*>(user_data);
      delete fd;
}

void *FalconDataManager::onClone( VMachine *vm, void *user_data )
{
      FalconData *fd = static_cast<FalconData*>(user_data);
      return fd->clone();
}


// Define a unique data manager valid for all the engine
FalconDataManager core_falcon_data_manager;

}

/* end of falcondata.cpp */
