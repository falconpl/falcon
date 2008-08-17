/*
   FALCON - The Falcon Programming Language.
   FILE: intcomp.h

   Complete encapsulation of an incremental interactive compiler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Aug 2008 11:10:24 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_INTCOMP_H
#define FALCON_INTCOMP_H

#include <falcon/setup.h>
#include <falcon/compiler.h>
#include <falcon/srclerxer.h>
#include <falcon/module.h>
#include <falcon/runtime.h>
#include <falcon/flcloader.h>
#include <falcon/vm.h>

namespace Falcon
{

class InteractiveCompiler: public Compiler
{
   VMachine m_vm;
   FlexyModule *flexyMod;

public:

   InteractiveCompiler(

   typedef enum {
      nothing,
      decl,
      statement,
      error
   } ret_type;

   ret_type compileNext( Stream *input );

};

}

#endif

/* end of intcomp.h */
