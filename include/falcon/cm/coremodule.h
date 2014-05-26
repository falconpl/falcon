/*
   FALCON - The Falcon Programming Language.
   FILE: coremodule.h

   Core module - basic Falcon functions usually available.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 12:25:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_COREMODULE_H
#define	_FALCON_COREMODULE_H

#include <falcon/module.h>

namespace Falcon {
/** The Core Module.
 
 This module contains the standard functions that are usually available to scripts.
 
 The functions in this module are still a little above language-level (or builtin)
 functions, which are resolved immediately at compiler level and are provided by
 the engine.

 In this way, embedding applications can chose to create other versions of
 the core module, or removing it altogether, without the need to re-introduce
 very basic and language-bound functions. In fact, while functions like printl
 may be completely out of context in an embedded script having to drive an
 application, max(), int() and toString() are definitely  part of the language
 standard and should always be present as pre-defined.
 
 */
class FALCON_DYN_CLASS CoreModule: public Module
{
public:
   CoreModule();

   Class* clsTextWriter() const { return m_ctw; }
   Class* clsTextReader() const { return m_ctr; }

private:
   Class* m_ctw;
   Class* m_ctr;
};

}

#endif	/* _FALCON_COREMODULE_H */

/* end of coremodule.h */
