/*
   FALCON - The Falcon Programming Language.
   FILE: coremodule.cpp

   Core module -- main file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 12:25:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/coremodule.cpp"

#include <falcon/cm/coremodule.h>

#include <falcon/cm/compile.h>
#include <falcon/cm/iff.h>
#include <falcon/cm/inspect.h>
#include <falcon/cm/print.h>
#include <falcon/cm/uri.h>
#include <falcon/cm/path.h>
#include <falcon/cm/textstream.h>
#include <falcon/cm/textwriter.h>
#include <falcon/cm/textreader.h>
#include <falcon/cm/datawriter.h>
#include <falcon/cm/datareader.h>
#include <falcon/cm/parallel.h>
#include <falcon/cm/stdfunctions.h>

// the standard error classes
#include <falcon/errorclasses.h>

#include <falcon/engine.h>

namespace Falcon {

CoreModule::CoreModule():
   Module("core")
{
   static ClassStream* classStream = static_cast<ClassStream*>(
            Engine::instance()->streamClass());
   
   *this
      // Standard functions
      << new Ext::Compile
      << new Ext::FuncPrintl
      << new Ext::FuncPrint
      << new Ext::Inspect
      << new Ext::Iff
      << new Ext::Function_epoch
      << new Ext::Function_seconds
      << new Ext::Function_sleep
      
      // Standard classes
      << new Ext::ClassURI
      << new Ext::ClassPath
      << new Ext::ClassParallel
      << new Ext::ClassTextStream( classStream )
      << new Ext::ClassTextWriter( classStream )
      << new Ext::ClassTextReader( classStream )
      << new Ext::ClassDataWriter( classStream )
      << new Ext::ClassDataReader( classStream )
      ;
}

}

/* end of coremodule.cpp */
