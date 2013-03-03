/*
   FALCON - The Falcon Programming Language.
   FILE: int_mode.h

   Falcon compiler and interpreter - interactive mode.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 23 Mar 2009 18:57:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_INT_MODE_H
#define FALCON_INT_MODE_H

#include "app.h"

namespace Falcon {

class FalconApp;

class IntMode
{
public:
   IntMode( FalconApp* owner );
   void run();

private:
   bool read_line( const String& prompt, String &line );
   FalconApp *m_owner;
   
   VMachine m_vm;
};

} // namespace Falcon

#endif

/* end of int_mode.h */
