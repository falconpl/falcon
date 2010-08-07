/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_mod.cpp

   The SDL True Type binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Paul Davey
   Begin: Tue, 12 Aug 2009 00:06:56 +1100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL True Type binding support module - module specific extensions.
*/

#include <falcon/vm.h>
#include <falcon/membuf.h>
#include "glext_mod.h"

#include <SDL_opengl.h>


/*# @beginmodule sdlopengl */

namespace Falcon {
namespace Ext {

//=======================================
// Quit carrier
//
GLExtQuitCarrier::~GLExtQuitCarrier()
{
   
}

}
}

/* end of sdlopengl_mod.cpp */
