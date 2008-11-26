/*
   FALCON - The Falcon Programming Language.
   FILE: userdata.cpp

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

#include <falcon/userdata.h>
#include <falcon/cobject.h>

namespace Falcon {

UserDataManager::UserDataManager( bool bNeedCache, UserData *model ):
   FalconDataManager( model ),
   m_bNeedCache(bNeedCache)
{}

UserDataManager::~UserDataManager()
{}

bool UserDataManager::onSetProperty( CoreObject *owner, void *user_data, const String &propname, const Item &property )
{
   if( user_data != 0 )
   {
      UserData *ud = (UserData *)user_data;
      ud->setProperty( owner, propname, property );
      return true; // ud->setProperty raises on need.
   }
   return false;
}

bool UserDataManager::onGetProperty( CoreObject *owner, void *user_data, const String &propname, Item &property )
{
   if ( user_data != 0 )
   {
      UserData *ud = (UserData *)user_data;
      ud->getProperty( owner, propname, property );
      return true; // ud->setProperty raises on need.
   }
   return false;
}

UserDataManager core_user_data_manager_cacheful( true );
UserDataManager core_user_data_manager_cacheless( false );

}

/* end of userdata.cpp */
