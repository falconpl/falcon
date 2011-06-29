/*
   FALCON - The Falcon Programming Language.
   FILE: coremodule.cpp

   Function objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 12:25:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/cm/coremodule.h>
#include <falcon/cm/compare.h>
#include <falcon/cm/describe.h>
#include <falcon/cm/len.h>
#include <falcon/cm/minmax.h>
#include <falcon/cm/print.h>
#include <falcon/cm/tostring.h>
#include <falcon/cm/typeid.h>

#include <falcon/corestring.h>
#include <falcon/corenil.h>
#include <falcon/corebool.h>
#include <falcon/coreint.h>
#include <falcon/corenumeric.h>
#include <falcon/coredict.h>
#include <falcon/corearray.h>

namespace Falcon {

CoreModule::CoreModule():
   Module("core", true)
{
   *this
      << new Ext::Compare
      << new Ext::Describe
      << new Ext::Len
      << new Ext::FuncPrintl
      << new Ext::FuncPrint
      << new Ext::Min
      << new Ext::Max
      << new Ext::ToString
      << new Ext::TypeId
      
      << new CoreNil
      << new CoreBool
      << new CoreInt
      << new CoreNumeric
      << new CoreArray
      << new CoreDict
      << new CoreString
      ;
}

}

/* end of coremodule.cpp */
