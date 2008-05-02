/*
   FALCON - The Falcon Programming Language.
   FILE: engine.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mag 6 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Main embedding inclusion file.

   The embedding application should include this file to include all the necessary
   files.
*/

#ifndef flc_engine_H
#define flc_engine_H

// basic functionalities
#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/memory.h>

// Falcon item system
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/cclass.h>
#include <falcon/cobject.h>
#include <falcon/lineardict.h>
#include <falcon/pagedict.h>

// Falcon String helpers
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>

// Falcon stream helpers
#include <falcon/stdstreams.h>
#include <falcon/uri.h>

// error system
#include <falcon/deferrorhandler.h>

// compiler and builder
#include <falcon/compiler.h>
#include <falcon/flcloader.h>
#include <falcon/runtime.h>

// main VM and helpers
#include <falcon/module.h>
#include <falcon/vm.h>

// Environmental support
#include <falcon/core_ext.h>
// #include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/stringstream.h>
#include <falcon/rosstream.h>

// Special types
#include <falcon/genericvector.h>
#include <falcon/genericlist.h>
#include <falcon/genericmap.h>
#include <falcon/timestamp.h>

// Engine dll and initialization
#include <falcon/enginedata.h>

#endif

/* end of engine.h */
