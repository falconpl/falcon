/*
   FALCON - The Falcon Programming Language.
   FILE: sdl_event_ext.cpp

   Binding for SDL event subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 22 Mar 2008 20:29:06 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Binding for SDL event subsystem.
*/

#include <falcon/vm.h>
#include <falcon/transcoding.h>
#include <falcon/fstream.h>
#include <falcon/lineardict.h>
#include <falcon/autocstring.h>
#include <falcon/membuf.h>

#include "sdl_ext.h"
#include "sdl_mod.h"

#include <SDL.h>

namespace Falcon {
namespace Ext {

void declare_events( Module *self )
{
}

}
}
