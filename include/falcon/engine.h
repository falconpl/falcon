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

// OS signal handling
#include <falcon/signals.h>

// Global engine functions and variables
#include <falcon/globals.h>
#include <falcon/transcoding.h>

// Falcon item system
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/coredict.h>
#include <falcon/cclass.h>
#include <falcon/cclass.h>
#include <falcon/coreobject.h>
#include <falcon/corecarrier.h>
#include <falcon/falconobject.h>
#include <falcon/crobject.h>
#include <falcon/reflectobject.h>
#include <falcon/lineardict.h>
#include <falcon/pagedict.h>
#include <falcon/membuf.h>
#include <falcon/continuation.h>

// Falcon String helpers
#include <falcon/autocstring.h>
#include <falcon/autowstring.h>

// Falcon stream helpers
#include <falcon/stdstreams.h>
#include <falcon/fstream.h>
#include <falcon/vfsprovider.h>
#include <falcon/uri.h>

// compiler and builder
#include <falcon/compiler.h>
#include <falcon/modloader.h>
#include <falcon/runtime.h>

// main VM and helpers
#include <falcon/module.h>
#include <falcon/flexymodule.h>
#include <falcon/vm.h>
#include <falcon/garbagelock.h>
#include <falcon/vmevent.h>
#include <falcon/attribmap.h>

// Environmental support
#include <falcon/core_ext.h>
// #include <falcon/error.h>
#include <falcon/stream.h>
#include <falcon/stringstream.h>
#include <falcon/rosstream.h>
#include <falcon/streambuffer.h>

// Special types
#include <falcon/genericvector.h>
#include <falcon/genericlist.h>
#include <falcon/genericmap.h>
#include <falcon/timestamp.h>

// Falcon specific object user_data
#include <falcon/falcondata.h>
#include <falcon/sequence.h>
#include <falcon/iterator.h>


#endif

/* end of engine.h */
