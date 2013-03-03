/*
   FALCON - The Falcon Programming Language.
   FILE: include.h

   Falcon core module -- Dynamic compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 05 Feb 2013 17:34:06 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_INCLUDE_H
#define FALCON_CORE_INCLUDE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/pstep.h>

namespace Falcon {
class Module;
class ItemDict;

namespace Ext {


/*#
   @function include
   @brief Dynamically loads a module as a plugin.
   @param file The relative filename of the module.
   @optparam inputEnc Input encoding.
   @optparam path A string of ';' separated search paths.
   @optparam symDict Symbols to be queried (or nil).

   @raise IoError if the module cannot be found or loaded.

   A module indicated by filename is compiled, loaded and linked in the
   running Virtual Machine. The inclusion is relative to the current
   path, be it set in the script, in the current embedding application
   or in the falcon command line interpreter. It is possible to use
   a path relative to the current script path by using the scriptPath
   variable.

   If a dictionary of symbols to be queried is @b not provided, the module
   is loaded and its main code, if present, is executed.

   If @b symDict is provided, its keys are strings which refer to symbol names
   to be searched in the loaded module. If present, the entry are filled with
   symbols coming from the loaded module. When @b symDict is provided, the
   linked module won't have its main code executed (it is possible to execute
   it at a later time adding "__main__" to the searched symbols). If a symbol
   is not found, its entry in the dictionary will be set to nil. When loaded
   this way, the export requests in the loaded module are @b not honored (import/from
   semantic).

   The @b compiler Feather module provides a more complete interface to dynamic
   load and compilation, but this minimal dynamic load support is provided at
   base level for flexibility.
*/

class Function_include: public ::Falcon::Function
{
public:
   Function_include();
   virtual ~Function_include();
   virtual void invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 pCount = 0 );

   private:
      FALCON_DECLARE_INTERNAL_PSTEP_OWNED( ModLoaded, Function_include );
      void getModSymbols( ItemDict& syms, Module *module );
};

}
}

#endif	

/* end of include.h */
