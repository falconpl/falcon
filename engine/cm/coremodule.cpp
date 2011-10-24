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
#include <falcon/cm/minmax.h>
#include <falcon/cm/print.h>
#include <falcon/cm/tostring.h>
#include <falcon/cm/typeid.h>
#include <falcon/cm/clone.h>
#include <falcon/cm/uri.h>
#include <falcon/cm/path.h>
#include <falcon/cm/stream.h>

#include <falcon/classes/classstring.h>
#include <falcon/classes/classnil.h>
#include <falcon/classes/classbool.h>
#include <falcon/classes/classint.h>
#include <falcon/classes/classnumeric.h>
#include <falcon/classes/classdict.h>
#include <falcon/classes/classarray.h>
#include <falcon/classes/classrange.h>
#include <falcon/classes/classreference.h>
#include <falcon/flexyclass.h>
#include <falcon/prototypeclass.h>
#include <falcon/metaclass.h>

// the standard error classes
#include <falcon/errorclasses.h>


namespace Falcon {

CoreModule::CoreModule():
   Module("core")
{
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
      
      // Standard classes
      << new Ext::ClassURI
      << new Ext::ClassPath      
      << new Ext::ClassStream
      ;
}

}

/* end of coremodule.cpp */
