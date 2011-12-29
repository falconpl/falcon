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
#include <falcon/cm/compare.h>
#include <falcon/cm/describe.h>
#include <falcon/cm/len.h>
#include <falcon/cm/baseclass.h>
#include <falcon/cm/inspect.h>
#include <falcon/cm/classname.h>
#include <falcon/cm/minmax.h>
#include <falcon/cm/print.h>
#include <falcon/cm/tostring.h>
#include <falcon/cm/typeid.h>
#include <falcon/cm/clone.h>
#include <falcon/cm/uri.h>
#include <falcon/cm/path.h>
#include <falcon/cm/storer.h>
#include <falcon/cm/restorer.h>
#include <falcon/cm/stream.h>
#include <falcon/cm/textstream.h>
#include <falcon/cm/textwriter.h>
#include <falcon/cm/datareader.h>
#include <falcon/cm/datawriter.h>

// the standard error classes
#include <falcon/errorclasses.h>



namespace Falcon {

CoreModule::CoreModule():
   Module("core")
{
   Ext::ClassStream* classStream = new Ext::ClassStream;
   
   *this
      // Standard functions
      << new Ext::Compare
      << new Ext::Describe
      << new Ext::Len
      << new Ext::FuncPrintl
      << new Ext::FuncPrint
      << new Ext::Min
      << new Ext::Max
      << new Ext::ToString
      << new Ext::TypeId
      << new Ext::Clone
      << new Ext::ClassName
      << new Ext::BaseClass
      << new Ext::Inspect
      
      // Standard classes
      << new Ext::ClassURI
      << new Ext::ClassPath
      << new Ext::ClassRestorer
      << new Ext::ClassStorer
      << classStream
      << new Ext::ClassTextStream( classStream )
      << new Ext::ClassTextWriter( classStream )
      << new Ext::ClassDataWriter()
      << new Ext::ClassDataReader()
      ;
}

}

/* end of coremodule.cpp */
