/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_mod.h

   The SDL True Type binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:11:06 +0100
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
   The SDL True Type binding support module - module specific extensions.
*/

#ifndef FALCON_SDLTTF_MOD
#define FALCON_SDLTTF_MOD

#include <falcon/setup.h>
#include <falcon/suserdata.h>
#include <falcon/error.h>

#include <SDL_ttf.h>

#define FALCON_TTF_ERROR_BASE 2120

namespace Falcon{
namespace Ext{

/** Automatic quit system. */
class TTFQuitCarrier: public UserData
{
public:
   TTFQuitCarrier() {}
   ~TTFQuitCarrier();
};

class TTFFontCarrier: public UserData
{
public:
   TTF_Font *m_font;

   TTFFontCarrier( TTF_Font *font ):
      m_font( font )
   {}

   ~TTFFontCarrier();
};

}
}
#endif

/* end of sdlttf_mod.h */
