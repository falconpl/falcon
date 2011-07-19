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

#include <falcon/classstring.h>
#include <falcon/classnil.h>
#include <falcon/classbool.h>
#include <falcon/classint.h>
#include <falcon/classnumeric.h>
#include <falcon/classdict.h>
#include <falcon/classarray.h>
#include <falcon/flexyclass.h>
#include <falcon/prototypeclass.h>
#include <falcon/metaclass.h>


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
      
      << new ClassNil
      << new ClassBool
      << new ClassInt
      << new ClassNumeric
      << new ClassArray
      << new ClassDict
      << new ClassString
      << new FlexyClass
      << new PrototypeClass
      << new MetaClass
      ;
}

}

/* end of coremodule.cpp */
