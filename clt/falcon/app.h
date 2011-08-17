/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.h

   Falcon compiler and interpreter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 23 Mar 2009 18:57:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Main Falcon.
*/

#ifndef FALCON_CLT_H
#define FALCON_CLT_H

#include <falcon/falcon.h>

#include "options.h"
#include "int_mode.h"

namespace Falcon {

class FalconApp: public Falcon::Application
{

public:
   int m_exitValue;
   FalconOptions m_options;
   
   FalconApp();
   
   void guardAndGo( int argc, char* argv[] );
   void interactive();
   void launch( const String& script );
};

} // namespace Falcon

#endif

/* end of falcon.h */
