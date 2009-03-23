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
   Options storage for falcon compiler.
*/

#ifndef FALCON_CLT_H
#define FALCON_CLT_H

#include <falcon/sys.h>
#include <falcon/setup.h>
#include <falcon/common.h>
#include <falcon/compiler.h>
#include <falcon/genhasm.h>
#include <falcon/gencode.h>
#include <falcon/gentree.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/modloader.h>
#include <falcon/runtime.h>
#include <falcon/core_ext.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/memory.h>
#include <falcon/transcoding.h>
#include <falcon/stream.h>
#include <falcon/fstream.h>
#include <falcon/stringstream.h>
#include <falcon/stdstreams.h>
#include <falcon/fassert.h>
#include <falcon/intcomp.h>
#include <falcon/streambuffer.h>
#include <options.h>

void read_line( Stream *in, String &line, uint32 maxSize );
void interactive_mode( ModuleLoader *loader, Module *core );

#endif

/* end of falcon.h */
