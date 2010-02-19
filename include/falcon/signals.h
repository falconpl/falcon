/*
   FALCON - The Falcon Programming Language.
   FILE: signals.h

   OS signal handling
   -------------------------------------------------------------------
   Author: Jan Dvorak
   Begin: 2010-02-19

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_SIGNALS_H
#define FALCON_SIGNALS_H

#include <falcon/setup.h>

#ifndef FALCON_SYSTEM_WIN
# include <falcon/signals_posix.h>
#endif

namespace Falcon {

/** Block OS signals in this thread. */
void FALCON_DYN_SYM BlockSignals();

/** Unblock OS signals in this thread. */
void FALCON_DYN_SYM UnblockSignals();

}

#endif

// vim: et ts=3 sw=3 :
/* end of signals.h */
