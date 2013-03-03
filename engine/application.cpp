/*
   FALCON - The Falcon Programming Language.
   FILE: application.cpp

   Utility to create embedding applications.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 13:14:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/application.h>
#include <falcon/engine.h>
#include <stdlib.h>

namespace Falcon {

bool Application::m_bUnique = false;

Application::Application()
{
   if (m_bUnique)
   {
      abort();
   }

   m_bUnique = true;
   Engine::init();
}

Application::~Application()
{
   Engine::shutdown();
}

}

/* end of application.cpp */

