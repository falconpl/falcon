/*
   FALCON - The Falcon Programming Language.
   FILE: sdlttf_mod.h

   The SDL True Type binding support module - module specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Mar 2008 23:11:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The SDL True Type binding support module - module specific extensions.
*/

#ifndef FALCON_SDLTTF_MOD
#define FALCON_SDLTTF_MOD

#include <falcon/setup.h>
#include <falcon/falcondata.h>
#include <falcon/error.h>

extern "C" {
   #include <SDL_ttf.h>
}

#define FALCON_TTF_ERROR_BASE 2120

namespace Falcon{
namespace Ext{

/** Automatic quit system. */
class TTFQuitCarrier: public FalconData
{
public:
   TTFQuitCarrier() {}
   virtual ~TTFQuitCarrier();

   virtual void gcMark( VMachine* ) {}
   virtual FalconData* clone() const { return 0; }
};

class TTFFontCarrier: public FalconData
{
public:
   TTF_Font *m_font;

   TTFFontCarrier( TTF_Font *font ):
      m_font( font )
   {}

   virtual ~TTFFontCarrier();

   virtual void gcMark( VMachine* ) {}
   virtual FalconData* clone() const { return 0; }
};

}
}
#endif

/* end of sdlttf_mod.h */
